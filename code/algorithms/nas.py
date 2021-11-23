import copy
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph, Edge
from utils.logging import log_and_print, print_program, print_program_dict
from utils.data_loader import flatten_batch
from algorithms.search_methods import few_shot_eval

import dsl
import pickle
import os

class NAS(ProgramLearningAlgorithm):

    def __init__(self, frontier_capacity=float('inf')):
        self.frontier_capacity = frontier_capacity


    ########################################
    # Training Process  ####################
    ###########################################################################################################################
    # directly train graph without consider architecture search
    def train_graph_model(self, graph, train_loader, train_config, device, lr_decay=1.0):
        # get train arguments
        model_lr = train_config['train_lr'] * lr_decay
        num_epoches = train_config['finetune_epoches']
        model_optim_fun = train_config['model_optim']
        lossfxn = train_config['lossfxn']

        log_and_print('learning rate: {}'.format(model_lr))

        output_type = graph.output_type
        output_size = graph.output_size
        batch_size = train_loader.batch_size

        # init optimizer
        model_params, _ = graph.get_model_params()
        model_optim = model_optim_fun(model_params, model_lr, weight_decay=train_config['model_weight_decay'])
        model_params_list = []
        for param_dict in model_params:
            model_params_list += param_dict['params']

        # prepare validation data
        validset = train_loader.validset

        # train
        best_model = None
        best_loss = None
        best_epoch = None
        for epoch in tqdm(range(num_epoches)):
            # store
            log_and_print('------------------------')
            log_and_print('training epoch: {}'.format(epoch))
            loss_store = {'valid_loss':[], 'model_loss':[]}

            # get new train set
            trainset = train_loader.get_batch_trainset()
            for batchidx in tqdm(range(len(trainset))):
                # update model parameters
                # prepare train data
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))
                true_vals = torch.tensor(flatten_batch(batch_output)).float().to(device)
                # TODO a little hacky, but easiest solution for now
                if isinstance(lossfxn, nn.CrossEntropyLoss):
                    true_vals = true_vals.long()

                # pass through graph
                train_predicted = graph.execute_graph(batch_input, output_type, output_size, device, clear_temp=False, cur_arch_train=True)

                # backprop
                ratio = 1.0
                loss = ratio * lossfxn(train_predicted, true_vals)
                model_optim.zero_grad()
                loss.backward()

                # debug
                if loss.cpu().item() != loss.cpu().item():
                    with torch.no_grad():
                        train_predicted_index = torch.mean(torch.mean(train_predicted, dim=-1).view(len(batch_input), -1), dim=-1)
                        indxs = torch.logical_and(train_predicted_index == train_predicted_index, torch.abs(train_predicted_index) < 50)
                        log_and_print('found NaN, totally {} rows'.format(indxs.shape[0]-torch.sum(indxs)))
                        graph.clear_graph_results()
                    # retrain
                    true_indxs = indxs.unsqueeze(1).repeat(1, len(batch_input[0])).view(-1)
                    valid_batch_input = [batch_input[idx_num] for idx_num, idx in enumerate(indxs) if idx]
                    train_predicted = graph.execute_graph(valid_batch_input, output_type, output_size, device, clear_temp=False, cur_arch_train=True)
                    loss = lossfxn(train_predicted, true_vals[true_indxs])
                    # graph.clear_graph_results()
                    model_optim.zero_grad()
                    loss.backward()

                model_optim.step()
                graph.clear_graph_results()

                # store
                loss_store['model_loss'].append(loss.cpu().data.item())

            # evaluate
            with torch.no_grad():
                metric = self.eval_graph(graph, validset, train_config['evalfxn'], train_config['num_labels'], device)

            # log
            log_and_print('model loss: {}'.format(np.mean(loss_store['model_loss'])))
            log_and_print('validation metric: {}'.format(metric))

            if best_model is None:
                best_loss = metric
                best_model = copy.deepcopy(graph)
                best_epoch = epoch
            elif metric <= best_loss:
                best_loss = metric
                best_model = copy.deepcopy(graph)
                best_epoch = epoch

        log_and_print('finish train\n')
        log_and_print('best epoch: {}'.format(best_epoch))

        return best_model


    # train the model
    def train_graph_search(self, graph, search_loader, train_config, device, valid=False, lr_decay=1.0, warmup=False, get_best=False):
        # get train arguments
        arch_lr = train_config['arch_lr'] * lr_decay
        model_lr = train_config['model_lr'] * lr_decay
        num_epoches = train_config['search_epoches']
        arch_optim_fun = train_config['arch_optim']
        model_optim_fun = train_config['model_optim']
        lossfxn = train_config['lossfxn']
        second_order = train_config['secorder']

        output_type = graph.output_type
        output_size = graph.output_size
        batch_size = search_loader.batch_size

        # init optimizer
        arch_params, arch_params_num = graph.get_arch_params()
        arch_optim = arch_optim_fun(arch_params, arch_lr, weight_decay=train_config['arch_weight_decay'])
        arch_params_list = arch_params

        model_params, model_params_num = graph.get_model_params()
        # model_optim = model_optim_fun(model_params, model_lr, weight_decay=1e-4)
        model_optim = model_optim_fun(model_params, model_lr, weight_decay=train_config['model_weight_decay'])
        model_params_list = []
        for param_dict in model_params:
            model_params_list += param_dict['params']

        log_and_print('number of architecture parameters {}'.format(arch_params_num))
        log_and_print('number of model parameters {}'.format(model_params_num))
        log_and_print('ratio between arch/model parameter is: {}'.format(float(arch_params_num)/model_params_num))
        log_and_print('learning rate: {} | {}'.format(arch_lr, model_lr))

        # get new validation set
        best_metric = 10
        best_graph = copy.deepcopy(graph)
        cur_valid_id = 0
        validset = search_loader.get_batch_validset()
        # start train
        for epoch in tqdm(range(num_epoches)):
            # store
            log_and_print('------------------------')
            log_and_print('training epoch: {}'.format(epoch))
            loss_store = {'arch_loss':[], 'model_loss':[], 'valid_loss':[]}

            # store graph for debug
            pickle.dump(graph, open(os.path.join(train_config['save_path'], "graph_temp.p"), "wb"))

            # get new train set
            trainset = search_loader.get_batch_trainset()

            # train and search
            for batchidx in tqdm(range(len(trainset))):
                #########################################
                # update architecture parameters first  #
                #########################################
                # prepare validation data
                if cur_valid_id == len(validset):
                    cur_valid_id = 0
                    validset = search_loader.get_batch_validset()
                
                # not execute during warmup
                if not warmup:
                    # take value
                    validation_input, validation_output = map(list, zip(*validset[cur_valid_id]))
                    cur_valid_id += 1
                    validation_true_vals = torch.tensor(flatten_batch(validation_output)).float().to(device)
                    # TODO a little hacky, but easiest solution for now
                    if isinstance(lossfxn, nn.CrossEntropyLoss):
                        validation_true_vals = validation_true_vals.long()

                # prepare train data
                batch_input, batch_output = map(list, zip(*trainset[batchidx]))
                true_vals = torch.tensor(flatten_batch(batch_output)).float().to(device)
                # TODO a little hacky, but easiest solution for now
                if isinstance(lossfxn, nn.CrossEntropyLoss):
                    true_vals = true_vals.long()

                if not second_order and not warmup:
                    ratio = None
                    loss = self.forward_backward_check(graph, lossfxn, arch_optim,\
                                                validation_input, validation_true_vals, output_type, output_size, device,\
                                                clear_temp=True, cur_arch_train=True, clip=True, clip_params=arch_params_list, \
                                                optim_options=[model_optim], ratio=ratio)
                    if loss is None:
                        pdb.set_trace()
                        if get_best:
                            return None
                        else:
                            return False

                elif not warmup:
                    # zero grad
                    arch_optim.zero_grad()
                    # second-order
                    self.second_order(graph, batch_input, true_vals, validation_input, validation_true_vals, train_config, lr_decay, device, eta=model_lr)
                    # update parameters
                    arch_optim.step()
                    graph.clear_graph_results()

                    # test loss
                    with torch.no_grad():
                        valid_predicted = graph.execute_graph(validation_input, output_type, output_size, device, clear_temp=True, cur_arch_train=True)
                        loss = lossfxn(valid_predicted, validation_true_vals)

                # store
                if warmup:
                    loss_store['arch_loss'].append(math.inf)
                else:
                    loss_store['arch_loss'].append(loss.cpu().data.item())

                ###########################
                # update model parameters #
                ###########################
                ratio = None
                loss = self.forward_backward_check(graph, lossfxn, model_optim,\
                                            batch_input, true_vals, output_type, output_size, device, \
                                            clear_temp=True, cur_arch_train=False, clip=True, clip_params=model_params_list, \
                                            optim_options=[arch_optim], ratio=ratio)
                if loss is None:
                    if get_best:
                        return None
                    else:
                        return False

                # store
                loss_store['model_loss'].append(loss.cpu().data.item())

            # log
            log_and_print('architecture loss: {}'.format(np.mean(loss_store['arch_loss'])))
            log_and_print('model loss: {}'.format(np.mean(loss_store['model_loss'])))
            
            # evaluate
            if valid:
                with torch.no_grad():
                    validset_total = search_loader.validset
                    metric = self.eval_graph(graph, validset_total, train_config['evalfxn'], train_config['num_labels'], device)
                    if metric < best_metric:
                        best_metric = metric
                        best_graph = copy.deepcopy(graph)

        if get_best:
            return best_graph
        else:
            return True


    # forward and backward with None check
    def forward_backward_check(self, graph, lossfxn, optim, batch_data, label_data, output_type, output_size, device, \
                                clear_temp=True, cur_arch_train=False, clip=False, clip_params=None, manual_loss=None, optim_options=None, ratio=None):
        # zero grad
        if optim is not None:
            optim.zero_grad()

        # forward
        predicted_data = graph.execute_graph(batch_data, output_type, output_size, device, clear_temp=False, cur_arch_train=cur_arch_train)
        loss = lossfxn(predicted_data, label_data)

        # debug
        if ratio is not None:
            loss = loss * ratio
        if manual_loss is not None:
            loss += manual_loss

        loss.backward()
        early_cut = loss.cpu().item() != loss.cpu().item() or loss.cpu().item() > 100
        if clip:
            assert clip_params is not None
            clip_norm = nn.utils.clip_grad_norm_(clip_params, 0.25)
            # trace gradient if get NaN grad (consider trace grad and early cut if necessary)
            if clip_norm.item() != clip_norm.item():
                for param in clip_params:
                    if param.grad is None:
                        continue
                    grad_mean = torch.mean(param.grad).item()
                    if grad_mean != grad_mean:
                        param.grad = None
                clip_norm = nn.utils.clip_grad_norm_(clip_params, 0.25)
                assert clip_norm.item() == clip_norm.item()

        if early_cut:
            log_and_print('get Nan, delete invalid branch')
            # delete Nan programs
            with torch.no_grad():
                graph.early_cut_nan(nan_max=float(100))
                graph.clear_graph_results()
            # retrain
            if optim is not None:
                optim.zero_grad()
            if optim_options is not None:
                for optim in optim_options:
                    optim.zero_grad()

            predicted_data = graph.execute_graph(batch_data, output_type, output_size, device, clear_temp=False, cur_arch_train=cur_arch_train)
            loss = lossfxn(predicted_data, label_data)
            if ratio is not None:
                loss = ratio * loss

            loss.backward()
            # clip
            early_cut = loss.cpu().item() != loss.cpu().item()
            if clip:
                clip_norm = nn.utils.clip_grad_norm_(clip_params, 0.25)
                if clip_norm.item() != clip_norm.item():
                    for param in clip_params:
                        grad_mean = torch.mean(param.grad).item()
                        if grad_mean != grad_mean:
                            param.grad = None
            # check again
            if early_cut:
                return None

        # update
        if optim is not None:
            optim.step()
        if clear_temp:
            graph.clear_graph_results()
        
        return loss
    ###########################################################################################################################
    ###########################################################################################################################

    ########################################
    # Second order (not use as cost)  ######
    ###########################################################################################################################
    # second-order update for architecture parameters
    def second_order(self, graph, batch_train, train_label, batch_valid, valid_label, train_config, lr_decay, device, eta):
        # get train arguments
        model_lr = train_config['model_lr'] * lr_decay
        lossfxn = train_config['lossfxn']
        weight_decay = train_config['weight_decay']

        output_type = graph.output_type
        output_size = graph.output_size

        # copy graph for archtecture parameter update
        unrolled_graph = copy.deepcopy(graph)
        model_params, _ = unrolled_graph.get_model_params()
        arch_params, _ = unrolled_graph.get_arch_params()
        # prepare for gradient clip
        model_params_list = []
        for param_dict in model_params:
            model_params_list.append(param_dict['params'])

        # get next updated model
        self.forward_backward_check(unrolled_graph, lossfxn, None, \
                                    batch_train, train_label, output_type, output_size, device,
                                    clear_temp=True, cur_arch_train=False, clip=True, clip_params=model_params_list)

        for param_dict in model_params:
            for param_key in param_dict:
                param = param_dict[param_key]
                d_param = param.grad.data + weight_decay * param.data
                param.data = param.data - model_lr * d_param

        # init grad
        self._zero_model_grad(model_params)
        self._zero_arch_grad(arch_params)

        # update architecture with validation data
        self.forward_backward_check(unrolled_graph, lossfxn, None, \
                            batch_valid, valid_label, output_type, output_size, device,
                            clear_temp=True, cur_arch_train=False, clip=True, clip_params=model_params_list)
        
        arch_grad = [param.grad.data.clone() for param in arch_params]
        model_grad = self.extract_model_grad(model_params)

        # check grad (only for debug)
        for grad_dict in model_grad:
            for grad_key in grad_dict:
                grad = grad_dict[grad_key]

        # calculate update ratio (only for debug)
        grad_vector = []
        for grad_dict in model_grad:
            for grad_key in grad_dict:
                grad = grad_dict[grad_key]
                grad_vector.append(grad.view(-1))
        grad_norm = torch.norm(torch.cat(grad_vector))

        self._zero_model_grad(model_params)
        self._zero_arch_grad(arch_params)

        # get second-order gradient for architecture
        ori_arch_params, _ = graph.get_arch_params()
        ori_model_params, _ = graph.get_model_params()
        implicit_grads = self.hessian_vector_product(graph, ori_model_params, ori_arch_params, model_grad, \
                                                    lossfxn, batch_train, train_label, \
                                                    output_type, output_size, device, r=1e-2)

        for arch_grad_id, grad_vecs in enumerate(zip(arch_grad, implicit_grads)):
            arch_grad[arch_grad_id] = grad_vecs[0] - eta * grad_vecs[1]

        # update grad of architecture parameters
        for param, param_grad in zip(ori_arch_params, arch_grad):
            if param.grad is None:
                param.grad = Variable(param_grad)
            else:
                param.grad.data = param_grad


    def extract_model_grad(self, model_params):
        model_grad = [{param_key:param_dict[param_key].grad.data.clone() for param_key in param_dict} 
                                                                         for param_dict in model_params]
        return model_grad

    def _zero_model_grad(self, model_params):
        for param_dict in model_params:
            for param_key in param_dict:
                param = param_dict[param_key]
                param.grad.data.zero_()

    def _zero_arch_grad(self, arch_params):
        for param in arch_params:
            param.grad.data.zero_()

    def hessian_vector_product(self, graph, model_params, arch_param, model_grad, lossfxn, \
                                batch_input, input_label, output_type, output_size, \
                                device, r=1e-2):
        # prepare for gradient clip
        model_params_list = []
        for param_dict in model_params:
            model_params_list.append(param_dict['params'])

        # calculate update ratio
        grad_vector = []
        for grad_dict in model_grad:
            for grad_key in grad_dict:
                grad = grad_dict[grad_key]
                grad_vector.append(grad.view(-1))
        grad_norm = torch.norm(torch.cat(grad_vector))
        R = r / grad_norm

        # plus
        for param_dict_id, param_dict in enumerate(model_params):
            for param_key in param_dict:
                param = param_dict[param_key]
                param.data = param.data + R * model_grad[param_dict_id][param_key]

        self.forward_backward_check(graph, lossfxn, None, \
                                    batch_input, input_label, output_type, output_size, device,
                                    clear_temp=True, cur_arch_train=True, clip=True, clip_params=model_params_list)
        grad_plus = [param.grad.data.clone() for param in arch_param]

        # reset
        self._zero_arch_grad(arch_param)
        self._zero_model_grad(model_params)

        # minus
        for param_dict_id, param_dict in enumerate(model_params):
            for param_key in param_dict:
                param = param_dict[param_key]
                param.data = param.data - 2*R * model_grad[param_dict_id][param_key]

        self.forward_backward_check(graph, lossfxn, None, \
                                    batch_input, input_label, output_type, output_size, device,
                                    clear_temp=True, cur_arch_train=True, clip=True, clip_params=model_params_list)
        grad_minus = [param.grad.data.clone() for param in arch_param]

        # reset
        self._zero_arch_grad(arch_param)
        self._zero_model_grad(model_params)

        # calculate final grad
        grad_final = [(grad_plus_vec-grad_minus_vec) / (2*R) for grad_plus_vec, grad_minus_vec in zip(grad_plus, grad_minus)]

        # restore
        for param_dict_id, param_dict in enumerate(model_params):
            for param_key in param_dict:
                param = param_dict[param_key]
                param.data = param.data + R * model_grad[param_dict_id][param_key]

        return grad_final
    ###########################################################################################################################
    ###########################################################################################################################
    


    # evaluate the model
    def eval_graph(self, graph, test_set, eval_fun, num_labels, device):        
        output_type = graph.output_type
        output_size = graph.output_size

        # prepare test data
        test_input, test_output = map(list, zip(*test_set))
        test_true_vals = torch.tensor(flatten_batch(test_output)).float().to(device)
        test_true_vals = test_true_vals.long()
        # evaluation
        with torch.no_grad():
            # pass through graph
            test_predicted = graph.execute_graph(test_input, output_type, output_size, device)
            metric, additional_params = eval_fun(test_predicted, test_true_vals, num_labels=num_labels)

        # accuracy
        log_and_print("Validation score is: {:.4f}".format(metric))
        log_and_print("Average f1-score is: {:.4f}".format(1 - metric))
        log_and_print("Hamming accuracy is: {:.4f}".format(additional_params['hamming_accuracy']))

        return metric


    ########################################
    # Top-N preservatoin  ##################
    ###########################################################################################################################
    # search for current best program
    # top-down reduce edge and programs inside node
    def search_graph(self, graph, topN=1, max_depth=None, start_depth=0, penalty=0.0):
        assert topN > 0
        frontier = graph.get_nodes(start_depth)

        # start loop
        while len(frontier) > 0:
            # reduce edge and program for next
            new_frontier = []
            for node in frontier:
                if max_depth is not None and node.depth > max_depth:
                    continue

                graph.reduce_candidate(node, topN, penalty=penalty)
                # get programs
                programs = []
                for node_progs in node.prog_dict.values():
                    programs = programs + node_progs
                # move to next
                for program in programs:
                    for edge in program.get_submodules().values():
                        if isinstance(edge, Edge):
                            new_node = edge.get_next_node()
                            if new_node not in new_frontier:
                                new_frontier.append(new_node)

            frontier = new_frontier


    # step-by-step search graph base on f1 score
    def step_search_graph(self, graph, valid_set, eval_fun, num_labels, device, start_depth, end_depth, bcethre=None):
        bcethre = bcethre if bcethre is not None else 0.5
        cur_graph = copy.deepcopy(graph)
        # cur_graph = graph

        # prepare test data
        test_input, test_output = map(list, zip(*valid_set))
        test_true_vals = torch.tensor(flatten_batch(test_output)).float().to(device)

        with torch.no_grad():
            # search from root to children
            for cur_depth in range(start_depth, end_depth+1):
                # search
                cur_graph.step_search(cur_depth, eval_fun, test_input, test_true_vals, \
                                cur_graph.output_type, cur_graph.output_size, num_labels, device, bcethre=bcethre)
                # update
                last_nodes = cur_graph.get_nodes(cur_depth-1)
                for node in last_nodes:
                    cur_graph.reduce_candidate(node, 1)

        return cur_graph
    ###########################################################################################################################
    ###########################################################################################################################

    ########################################
    # few-shot search  ###########################
    ###########################################################################################################################
    # search with few-shot NAS method
    def few_shot_NAS_search(self, graph, search_loader, train_loader, train_config, device, penalty=0.0, time_thre=float('inf'), len_thre=float('inf')):
        log_and_print('few shot search with time thre to be {} and len thre to be {}'.format(time_thre, len_thre))
        # get frontiers
        frontier, terminal_frontier, total_time = few_shot_eval(graph, search_loader, train_loader, train_config, device, \
                                                    self.train_graph_model, self.train_graph_search, self.eval_graph, \
                                                    time_thre=time_thre, len_thre=len_thre, sel_num=train_config['sel_num'], penalty=penalty)
        # extract terminal graph
        start_time = time.time()
        validset = search_loader.validset
        for graph_info in frontier:
            _, cur_graph, _ = graph_info
            self.search_graph(cur_graph, topN=1, penalty=0.0)
            cur_cost = self.eval_graph(cur_graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
            terminal_frontier.append((cur_cost, cur_graph, None))
        # select the best
        best_cost, best_graph, _ = min(terminal_frontier, key=lambda x: x[0])
        select_time = time.time() - start_time
        best_graph.show_graph()
        log_and_print('best cost: {}'.format(best_cost))
        log_and_print('all train time: {}  all select time: {}'.format(total_time, select_time))

        return best_graph

    ########################################
    # A* search  ###########################
    ###########################################################################################################################
    # search the final path based on A-star
    def astar_search(self, graph, cur_metric, search_loader, train_loader, train_config, device, penalty=0.0):
        # check complete
        terminate = graph.check_terminate(graph, 0)
        if terminate:
            return graph
        # init
        best_cost = float('inf')
        best_graph = None
        frontier = [(cur_metric, graph, 0)]
        validset = search_loader.validset
        start_time = time.time()
        # A-star
        while len(frontier) > 0:
            _, cur_node, cur_depth = frontier.pop(0)
            # get children (left-to-right if multi-node in current depth)
            with torch.no_grad():
                # backward probability applied
                children, child_depth, all_cost = graph.get_children(cur_node, cur_depth, back_prob=True)
            log_and_print('current depth {} with totally {} children'.format(child_depth, len(children)))
            # get score
            for child_id, child_graph in enumerate(children):
                # debug
                child_graph.show_graph()
                terminate = child_graph.check_terminate(child_graph, child_depth)
                # train
                if terminate:
                    child_graph = self.train_graph_model(child_graph, search_loader, train_config, device, lr_decay=1.0)
                    cost = child_graph.get_cost(child_graph)
                else:
                    cost = all_cost[child_id]
                    child_graph = self.train_graph_search(child_graph, search_loader, train_config, device, valid=True, get_best=True)
                    if child_graph is None:
                        log_and_print('Nan appear')
                        continue
                # evaluate to get score
                child_cost = self.eval_graph(child_graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
                child_cost = child_cost + cost * penalty
                log_and_print('cost: {}'.format(cost))
                log_and_print('cost after combine: {}\n'.format(child_cost))
                # store
                if terminate and child_cost < best_cost:
                    best_cost = child_cost
                    best_graph = child_graph
                elif not terminate:
                    frontier.append((child_cost, child_graph, child_depth))

            # update frontier
            frontier.sort(key=lambda x: x[0])
            if len(frontier) > 0:
                log_and_print('current best {}  vs. frontier best {}'.format(best_cost, frontier[0][0]))
            else:
                log_and_print('current best {}  vs. empty frontier'.format(best_cost))
            while len(frontier) > 0 and frontier[-1][0] > best_cost:
                frontier.pop(-1)

        # print time
        log_and_print('total time for search {}'.format(time.time()-start_time))

        return best_graph

    ########################################
    # main for execution  ##################
    ###########################################################################################################################
    def run_specific(self, graph, search_loader, train_loader, train_config, device, start_depth=0, warmup=False, train_penalty=0.0):
        assert graph.root_node is not None
        cur_depth = start_depth
        max_depth = graph.max_depth
        penalty = train_config['penalty']

        # init nodes
        frontier = graph.get_nodes(cur_depth)

        # start loop
        cur_iter = 0
        lr_decay = 1.0
        time_store = {}
        while cur_iter < len(train_config['specific']):
            start = time.time()

            target_topN, target_depth, lr, num_epochs = train_config['specific'][cur_iter]
            log_and_print('current depth {}  after {}\n'.format(cur_depth, target_depth))

            # set value
            train_config['arch_lr'] = lr
            train_config['model_lr'] = lr
            train_config['search_epoches'] = num_epochs

            # check whether need to search
            if target_topN is not None:
                log_and_print('search for top {}'.format(target_topN))
                if target_topN == 'astar':
                    graph = self.astar_search(graph, float('inf'), search_loader, train_loader, train_config, device, penalty=penalty)
                    if not graph.check_neural(graph):
                        break
                elif target_topN == 'few_shot':
                    graph = self.few_shot_NAS_search(graph, search_loader, train_loader, train_config, device, penalty=penalty, \
                                                     time_thre=train_config['time_thre'], len_thre=train_config['len_thre'])
                    if not graph.check_neural(graph):
                        break
                else:
                    self.search_graph(graph, topN=target_topN, penalty=penalty*train_penalty)
                # evaluate entire graph
                testset = train_loader.testset
                self.eval_graph(graph, testset, train_config['evalfxn'], train_config['num_labels'], device)

            # extend
            assert target_depth >= cur_depth
            if target_depth > cur_depth:
                cell_depth = target_depth - cur_depth
                for node in frontier:
                    graph.build_next_cell(node, cell_depth, device, node_share=train_config['node_share'])
                if target_depth == max_depth:
                    graph.clean_candidate()
                # if warmup
                if warmup:
                    log_and_print('>warm up...')
                    train_config_backup = copy.deepcopy(train_config)
                    train_config['search_epoches'] = 3
                    self.train_graph_search(graph, search_loader, train_config, device, valid=True, lr_decay=lr_decay, warmup=True)
                    # evaluate warm up
                    testset = train_loader.testset
                    self.eval_graph(graph, testset, train_config['evalfxn'], train_config['num_labels'], device)
                    train_config = train_config_backup

            # train architecture and model
            log_and_print('> training...')
            # graph.reset_arch_params()
            graph.show_graph()
            if num_epochs != 0:
                self.train_graph_search(graph, search_loader, train_config, device, valid=True, lr_decay=lr_decay)

            # evaluate entire graph
            testset = train_loader.testset
            self.eval_graph(graph, testset, train_config['evalfxn'], train_config['num_labels'], device)

            # store intermediate graph
            pickle.dump(graph, open(os.path.join(train_config['save_path'], "graph_depth{}_iter{}.p".format(target_depth, cur_iter)), "wb"))
            graph.show_graph()
            log_and_print('\n\n')

            # execute time
            total_spend = time.time() - start
            time_store[cur_depth] = total_spend
            cur_depth = target_depth
            log_and_print('time spend: {} \n'.format(total_spend))

            # next loop
            frontier = graph.get_nodes(target_depth)

            # check complete program before extend
            if target_depth > cur_depth:
                complete = True
                for node in frontier:
                    programs = []
                    for prog_list in node.prog_dict.values():
                        programs += prog_list
                    non_complete = [1 for prog in programs if len(prog.get_submodules())!=0]
                    if len(non_complete) != 0:
                        complete = False
                        break
                
                if complete:
                    break

            cur_iter += 1

                    
        # start time
        start = time.time()
        # get best one
        self.search_graph(graph, topN=1, penalty=penalty*train_penalty)
        graph.show_graph()
        log_and_print('\n\n')
        # train and evaluate
        log_and_print('after search \n')
        best_graph = self.train_graph_model(graph, train_loader, train_config, device, lr_decay=lr_decay)
        # calculate time
        total_spend = time.time() - start
        log_and_print('time spend: {} \n'.format(total_spend))

        return best_graph

