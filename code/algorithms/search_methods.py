from utils.logging import log_and_print

import time
import torch
import random

import pdb

# evaluate search space with few-shot NAS method
def few_shot_eval(graph, search_loader, train_loader, train_config, device, train_method, search_method, eval_method,\
                  time_thre=float('inf'), len_thre=float('inf'), sel_num=1, penalty=0.0, order=False):
    # check complete
    terminate = graph.check_terminate(graph, 0)
    if terminate:
        return graph
    # init
    max_depth = graph.max_depth
    best_cost = float('inf')
    frontier = [(best_cost, graph, 0)]
    terminal_frontier = []
    validset = search_loader.validset
    start_time = time.time()
    # split sub-regions
    while len(frontier) > 0:
        # split all graphs in frontier
        new_frontier = []
        if order is None:
            split_depth = random.randint(0, max_depth-1)
        elif order:
            split_depth = 0
        else:
            split_depth = max_depth - 1
        log_and_print('--------------------current split depth: {} --------------------'.format(split_depth))
        log_and_print('current length to be: {}'.format(len(frontier)))
        for graph_info in frontier:
            _, cur_node, cur_depth = graph_info
            # get children (left-to-right if multi-node in current depth)
            with torch.no_grad():
                parents = [cur_node]
                term_children = []
                for sel_id in range(sel_num):
                    all_children = []
                    all_children_cost = []
                    for cur_p in parents:
                        # terminate
                        if cur_p.check_terminate(cur_p, 0):
                            log_and_print('terminate child meet')
                            term_children.append(cur_p)
                            continue
                        # split subregions
                        child_results = graph.get_children(cur_p, split_depth, back_prob=True)
                        prev_depth = [split_depth]
                        while child_results is None:
                            if order is None:
                                new_depth = random.randint(0, max_depth-1)
                                while new_depth in prev_depth:
                                    new_depth = random.randint(0, max_depth-1)
                            elif order:
                                new_depth = split_depth + 1
                            else:
                                new_depth = split_depth - 1
                            child_results = graph.get_children(cur_node, new_depth, back_prob=True)
                            prev_depth.append(new_depth)
                        if len(prev_depth) > 1:
                            log_and_print('-------new depth {}--------'.format(new_depth))
                        children, child_depth, all_cost = child_results
                        all_children += children
                        all_children_cost += all_cost
                        log_and_print('current depth {} with {} children'.format(child_depth, len(children)))
                    parents = all_children
            all_children += term_children
            log_and_print('totally {} children with {} terminate children'.format(len(all_children), len(term_children)))
            # pdb.set_trace()
            # get score
            for child_id, child_graph in enumerate(all_children):
                # debug
                # child_graph.show_graph()
                terminate = child_graph.check_terminate(child_graph, 0)
                # train
                if terminate:
                    child_graph = train_method(child_graph, search_loader, train_config, device, lr_decay=1.0)
                    cost = child_graph.get_cost(child_graph)
                else:
                    cost = all_children_cost[child_id]
                    child_graph = search_method(child_graph, search_loader, train_config, device, valid=True, get_best=True)
                    if child_graph is None:
                        log_and_print('Nan appear')
                        continue
                # evaluate to get score
                child_cost = eval_method(child_graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
                child_cost = child_cost + cost * penalty
                log_and_print('cost: {}'.format(cost))
                log_and_print('cost after combine: {}\n'.format(child_cost))
                # store
                if terminate:
                    terminal_frontier.append((child_cost, child_graph, child_depth))
                else:
                    new_frontier.append((child_cost, child_graph, child_depth))
        # sort and drop
        new_frontier.sort(key=lambda x: x[0])
        if len(new_frontier) > len_thre:
            new_frontier = new_frontier[:len_thre]
        frontier = new_frontier
        # break
        cur_time = time.time()-start_time
        if cur_time > time_thre:
            break
    # return
    return frontier, terminal_frontier, cur_time
