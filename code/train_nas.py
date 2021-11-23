import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

# import program_learning
from algorithms import NAS
from eval_test import test_set_eval
from program_graph import ProgramGraph
from utils.data_loader import CustomLoader
from utils.evaluation import label_correctness
from utils.logging import init_logging, log_and_print, print_program
from utils.loss import SoftF1LossWithLogits

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    # Args for experiment setup
    parser.add_argument('-t', '--trial', type=int, required=True,
                        help="trial ID")
    parser.add_argument('--exp_name', type=str, required=True,
                        help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # Args for data
    parser.add_argument('--train_data', type=str, required=True,
                        help="path to train data")
    parser.add_argument('--test_data', type=str, required=True, 
                        help="path to test data")
    parser.add_argument('--valid_data', type=str, required=False, default=None,
                        help="path to val data. if this is not provided, we sample val from train.")    
    parser.add_argument('--train_labels', type=str, required=True,
                        help="path to train labels")
    parser.add_argument('--test_labels', type=str, required=True, 
                        help="path to test labels")
    parser.add_argument('--valid_labels', type=str, required=False, default=None,
                        help="path to val labels. if this is not provided, we sample val from train.")     
    parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
                        help="input type of data")
    parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
                        help="output type of data")
    parser.add_argument('--input_size', type=int, required=True,
                        help="dimenion of features of each frame")
    parser.add_argument('--output_size', type=int, required=True, 
                        help="dimension of output of each frame (usually equal to num_labels")
    parser.add_argument('--num_labels', type=int, required=True, 
                        help="number of class labels")

    # Args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.0,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")

    # Args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation."+\
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50, 
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('-search_lr', '--search_learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits", "softf1"],
                        help="loss function for training")
    parser.add_argument('--f1double', action='store_true', required=False, default=False,
                        help='whether use double for soft f1 loss')
    parser.add_argument('--class_weights', type=str, required=False, default = None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--topN_select', type=int, required=False, default=1,
                        help="number of candidates remain in each search")
    parser.add_argument('--resume_graph', type=str, required=False, default=None,
                        help="resume graph from certain path if necessary")
    parser.add_argument('--sec_order', action='store_true', required=False, default=False,
                        help='whether use second order for architecture search')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help="manual seed")
    parser.add_argument('--finetune_epoch', type=int, required=False, default=None,
                        help='Epoch for finetuning the result graph.')
    parser.add_argument('--finetune_lr', type=float, required=0.01, default=None,
                        help='Epoch for finetuning the result graph.')

    # Args for algorithms
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=["mc-sampling", "mcts", "enumeration", "genetic", "astar-near", "iddfs-near", "rnn", 'nas'],
                        help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR"+
                        " (>1.0 prunes aggressively in this case)")    
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")
    parser.add_argument('--cell_depth', type=int, required=False, default=3,
                        help="max depth for each cell for nas algorithm")
    
    # Args for ablation setting
    parser.add_argument('--node_share', action='store_true', required=False, default=False,\
                        help='use node sharing is specific')
    parser.add_argument('--graph_unfold', action='store_true', required=False, default=False,\
                        help='use node sharing is specific')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if 'crim13' in args.exp_name:
        print('crim13 experiment')
        from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS
    elif 'basket' in args.exp_name:
        from dsl_basket import DSL_DICT, CUSTOM_EDGE_COSTS
    elif 'sk152' in args.exp_name:
        from dsl_sk152 import DSL_DICT, CUSTOM_EDGE_COSTS
    elif 'fly' in args.exp_name:
        from dsl_fly import DSL_DICT, CUSTOM_EDGE_COSTS
    else:
        print('undefined experiment')
        exit(0)

    # manual seed all random for debug
    log_and_print('random seed {}'.format(args.random_seed))
    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    full_exp_name = "{}_depth{}_{}_{:03d}".format(args.exp_name, args.max_depth, args.algorithm, args.trial)

    save_path = os.path.join(args.save_dir, full_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init log
    init_logging(save_path)
    log_and_print("Starting experiment {}\n".format(full_exp_name))

    train_data = np.load(args.train_data, allow_pickle=True)
    test_data = np.load(args.test_data, allow_pickle=True)
    valid_data = None
    train_labels = np.load(args.train_labels)
    test_labels = np.load(args.test_labels)
    valid_labels = None
    # assert train_data.shape[-1] == test_data.shape[-1] == args.input_size

    if args.valid_data is not None and args.valid_labels is not None:
        valid_data = np.load(args.valid_data, allow_pickle=True)
        valid_labels = np.load(args.valid_labels)

    # for model & architecture
    search_loader = CustomLoader(train_data, None, test_data, train_labels, valid_labels, test_labels, \
                                normalize=args.normalize, train_valid_split=args.train_valid_split, batch_size=args.batch_size, shuffle=False, \
                                by_label=(args.output_type=='atom'))
    batched_trainset = search_loader.get_batch_trainset()
    batched_validset = search_loader.get_batch_validset()

    log_and_print('data for architecture search')
    log_and_print('batch num of train: {}'.format(len(batched_trainset)))
    log_and_print('batch num of valid: {}'.format(len(batched_validset)))

    # for program train
    train_loader = CustomLoader(train_data, valid_data, test_data, train_labels, valid_labels, test_labels, \
                                normalize=args.normalize, train_valid_split=args.train_valid_split, batch_size=args.batch_size, shuffle=False,\
                                by_label=(args.output_type=='atom'))
    batched_prog_trainset = train_loader.get_batch_trainset()
    prog_validset = train_loader.get_batch_validset()
    testset = train_loader.testset

    log_and_print('data for architecture search')
    log_and_print('batch num of train: {}'.format(len(batched_prog_trainset)))
    log_and_print('batch num of valid: {}'.format(len(prog_validset)))
    log_and_print('total num of test: {}'.format(len(testset)))

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    log_and_print('with loss function: {}'.format(args.lossfxn))
    if args.class_weights is None:
        if args.lossfxn == "crossentropy":
            lossfxn = nn.CrossEntropyLoss()
        elif args.lossfxn == "bcelogits":
            lossfxn = nn.BCEWithLogitsLoss()
        elif args.lossfxn == "softf1":
            lossfxn = SoftF1LossWithLogits(double=args.f1double)
    else:
        class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')])
        if args.lossfxn == "crossentropy":
            lossfxn = nn.CrossEntropyLoss(weight = class_weights)
        elif args.lossfxn == "bcelogits":
            lossfxn = nn.BCEWithLogitsLoss(pos_weight = class_weights)
        elif args.lossfxn == "softf1":
            lossfxn = SoftF1LossWithLogits(weight = class_weights, double=args.f1double)

    if device != 'cpu':
        lossfxn = lossfxn.cuda()


    # specific: [[search N, train_depth], ...] (basketball)
    # basketball
    if 'basket' in args.exp_name:
        specific = [[None, 2, 0.02, 10], [4, 2, 0.01, 6], [3, 2, 0.001, 6], [2, 2, 0.001, 3], \
                    [None, 4, 0.02, 3], [4, 4, 0.01, 3], [3, 4, 0.001, 3], ["astar", 4, 0.02, args.neural_epochs]]
        if not args.graph_unfold:
            specific = [[None, 4, 0.02, 10], [4, 4, 0.01, 6], [3, 4, 0.001, 6], [2, 4, 0.001, 3], ["astar", 4, 0.02, args.neural_epochs]]
        train_config = {
            'node_share' : args.node_share,
            'arch_lr' : args.search_learning_rate,
            'model_lr' : args.search_learning_rate, 
            'train_lr' : args.learning_rate,
            'search_epoches' : args.neural_epochs,
            'finetune_epoches' : args.symbolic_epochs,
            'arch_optim' : optim.Adam,
            'model_optim' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels,
            'save_path' : save_path,
            'topN' : args.topN_select,
            'arch_weight_decay' : 1e-3,
            'model_weight_decay' : 1e-4,
            'secorder' : args.sec_order,
            'penalty' : args.penalty,
            'specific' : specific
        }
    # crim13
    elif 'crim13' in args.exp_name:
        specific = [[None, 2, 0.001, 8], [4, 2, 0.001, 3], [3, 2, 0.001, 3], [2, 2, 0.001, 3], \
                    [None, 4, 0.001, 5], [4, 4, 0.001, 3], [3, 4, 0.001, 3], [2, 4, 0.001, 0], ["astar", 4, 0.001, args.neural_epochs]]
        if not args.graph_unfold:
            specific = [[None, 4, 0.001, 8], [4, 4, 0.001, 3], [3, 4, 0.001, 3], [2, 4, 0.001, 3], ["astar", 4, 0.001, args.neural_epochs]]
        train_config = {
            'node_share' : args.node_share,
            'arch_lr' : args.search_learning_rate,
            'model_lr' : args.search_learning_rate, 
            'train_lr' : args.learning_rate,
            'search_epoches' : args.neural_epochs,
            'finetune_epoches' : args.symbolic_epochs,
            'arch_optim' : optim.Adam,
            'model_optim' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels,
            'save_path' : save_path,
            'topN' : args.topN_select,
            'arch_weight_decay' : 0,
            'model_weight_decay' : 0,
            'secorder' : args.sec_order,
            'penalty' : args.penalty,
            'specific' : [[None, 2, 0.001, 8], [4, 2, 0.001, 3], [3, 2, 0.001, 3], [2, 2, 0.001, 3], \
                    [None, 4, 0.001, 5], [4, 4, 0.001, 3], [3,4, 0.001, 3], [2, 4, 0.001, 0], ["astar", 4, 0.001, args.neural_epochs]]
        }
    # sk152
    elif 'sk152' in args.exp_name:
        specific = [[None, 2, 0.01, 5], [4, 2, 0.01, 2], [3, 2, 0.01, 2], [2, 2, 0.01, 2],\
                    [None, 4, 0.01, 3], [4, 4, 0.01, 2], [3, 4, 0.01, 2], [2, 4, 0.01, 2], ["astar", 4, 0.01, args.neural_epochs]]
        if not args.graph_unfold:
            specific = [[None, 4, 0.01, 5], [4, 4, 0.01, 2], [3, 4, 0.01, 2], [2, 4, 0.01, 2], ["astar", 4, 0.01, args.neural_epochs]]
        train_config = {
            'node_share' : args.node_share,
            'arch_lr' : args.search_learning_rate,
            'model_lr' : args.search_learning_rate, 
            'train_lr' : args.learning_rate,
            'search_epoches' : args.neural_epochs,
            'finetune_epoches' : args.symbolic_epochs,
            'arch_optim' : optim.Adam,
            'model_optim' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels,
            'save_path' : save_path,
            'topN' : args.topN_select,
            'arch_weight_decay' : 0,
            'model_weight_decay' : 0,
            'penalty' : args.penalty,
            'secorder' : args.sec_order,
            'specific' : [[None, 2, 0.01, 5], [4, 2, 0.01, 2], [3, 2, 0.01, 2], [2, 2, 0.01, 2],\
                    [None, 4, 0.01, 3], [4, 4, 0.01, 2], [3, 4, 0.01, 2], [2, 4, 0.01, 2], ["astar", 4, 0.01, args.neural_epochs]]
        }
    # fly
    elif 'fly' in args.exp_name:
        specific = [[None, 2, 0.001, 5], [4, 2, 0.0005, 3], [3, 2, 0.0005, 3], [2, 2, 0.0005, 3], \
                    [None, 4, 0.0005, 5], [4, 4, 0.0005, 3], [3,4, 0.0005, 3], ["astar", 4, 0.0005, args.neural_epochs]]
        if not args.graph_unfold:
            specific = [[None, 4, 0.001, 5], [4, 4, 0.0005, 3], [3, 4, 0.0005, 3], [2, 4, 0.0005, 3], ["astar", 4, 0.0005, args.neural_epochs]]
        train_config = {
            'node_share' : args.node_share,
            'arch_lr' : args.search_learning_rate,
            'model_lr' : args.search_learning_rate, 
            'train_lr' : args.learning_rate,
            'search_epoches' : args.neural_epochs,
            'finetune_epoches' : args.symbolic_epochs,
            'arch_optim' : optim.Adam,
            'model_optim' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels,
            'save_path' : save_path,
            'topN' : args.topN_select,
            'arch_weight_decay' : 0,
            'model_weight_decay' : 0,
            'penalty' : args.penalty,
            'secorder' : args.sec_order,
            'specific' : [[None, 2, 0.001, 5], [4, 2, 0.0005, 3], [3, 2, 0.0005, 3], [2, 2, 0.0005, 3], \
                    [None, 4, 0.0005, 5], [4, 4, 0.0005, 3], [3,4, 0.0005, 3], ["astar", 4, 0.0005, args.neural_epochs]]
        }
    else:
        print('undefined experiment')
        exit(0)

    # Initialize program graph
    if args.resume_graph is None:
        program_graph = ProgramGraph(DSL_DICT, args.input_type, args.output_type, args.input_size, args.output_size,
                                    args.max_num_units, args.min_num_units, args.max_depth, device, ite_beta=args.ite_beta)
        start_depth = 0
    else:
        assert os.path.isfile(args.resume_graph)
        program_graph = pickle.load(open(args.resume_graph, "rb"))
        program_graph.max_depth = args.max_depth
        start_depth = program_graph.get_current_depth()
        # start_depth = 3

    # Initialize algorithm
    algorithm = NAS(frontier_capacity=args.frontier_capacity)

    # Run program learning algorithm
    best_graph = algorithm.run_specific(program_graph,\
                                search_loader, train_loader,
                                train_config, device, start_depth=start_depth, warmup=False)

    best_program = best_graph.extract_program()

    # print program
    log_and_print("Best Program Found:")
    program_str = print_program(best_program)
    log_and_print(program_str)

    # Save best program
    pickle.dump(best_graph, open(os.path.join(save_path, "graph.p"), "wb"))

    # Evaluate best program on test set
    log_and_print('Before finetune')
    test_set_eval(best_program, testset, args.output_type, args.output_size, args.num_labels, device)
    log_and_print("ALGORITHM END \n\n")
    
    # Finetune
    if args.finetune_epoch is not None:
        if 'sk152' in args.exp_name:
            # for program train
            train_loader = CustomLoader(train_data, valid_data, test_data, train_labels, valid_labels, test_labels, \
                                        normalize=True, train_valid_split=args.train_valid_split, batch_size=args.batch_size, shuffle=False,\
                                        by_label=(args.output_type=='atom'))
            testset = train_loader.testset

        train_config = {
            'train_lr' : args.finetune_lr,
            'search_epoches' : args.neural_epochs,
            'finetune_epoches' : args.finetune_epochs,
            'model_optim' : optim.Adam,
            'lossfxn' : lossfxn,
            'evalfxn' : label_correctness,
            'num_labels' : args.num_labels,
            'save_path' : save_path,
            'topN' : args.topN_select,
            'arch_weight_decay' : 0,
            'model_weight_decay' : 0,
            'secorder' : args.sec_order
        }
        log_and_print('Finetune')
        # start time
        start = time.time()
        best_graph = algorithm.train_graph_model(best_graph, train_loader, train_config, device, lr_decay=1.0)
        # calculate time
        total_spend = time.time() - start
        log_and_print('finetune time spend: {} \n'.format(total_spend))
        # store
        pickle.dump(best_graph, open(os.path.join(save_path, "finetune_graph.p"), "wb"))

        # debug
        testset = train_loader.testset
        best_program = best_graph.extract_program()
        log_and_print('After finetune')
        test_set_eval(best_program, testset, args.output_type, args.output_size, args.num_labels, device)
        log_and_print("ALGORITHM END \n\n")