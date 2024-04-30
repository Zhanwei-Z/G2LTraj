import os
import argparse
import baseline
from EigenTrajectory import *
from utils import *
import torch
import random
import numpy as np
import json
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/eigentrajectory-gpgraphsgcn-eth.json", type=str, help="config file path")
    parser.add_argument('--tag', default="EigenTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for the model")
    parser.add_argument('--static_dist', default=None, type=str, help="static_dist")
    parser.add_argument('--v2', default=None, type=str, help="v2")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args)) 

    print("===== Configs =====")
    if args.test:
        args.cfg = "./checkpoints/" + args.cfg
    hyper_params = get_exp_config(args.cfg)
    if not args.test:
        if args.static_dist and args.v2:
            if float(args.static_dist) < 10:
                hyper_params.static_dist = float(args.static_dist)
            if float(args.v2) < 10:
                hyper_params.v2 = float(args.v2)
        hyper_params1 = dotdict_to_dict(hyper_params)
        checkpoint_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(checkpoint_dir+'hyper_params.json', 'w') as json_file:
            json.dump(hyper_params1, json_file)

    print_arguments(hyper_params)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    hook_func = DotDict({"model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
                         "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
                         "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook})

    ModelTrainer = getattr(trainer, *[s for s in trainer.__dict__.keys() if hyper_params.baseline in s.lower()])
    trainer = ModelTrainer(base_model=PredictorModel, model=EigenTrajectory, hook_func=hook_func,
                           args=args, hyper_params=hyper_params)

    if not args.test:
        trainer.init_descriptor()
        trainer.fit()
    else:
        trainer.load_model(filename='model_best.pth')
        print("Testing...model_best")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

        trainer.load_model(filename='model_best_test.pth')
        print("Testing...model_best_test")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results_test.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

        trainer.load_model(filename='model_best_fde.pth')
        print("Testing...model_best_fde")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results_fde.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

        trainer.load_model(filename='model_best_test_fde.pth')
        print("Testing...model_best_test_fde")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results_test_fde.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

        trainer.load_model(filename='model_best_test_tcc.pth')
        print("Testing...model_best_test_tcc")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results_test_tcc.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

        trainer.load_model(filename='model_best_test_col.pth')
        print("Testing...model_best_test_col")
        results = trainer.test()
        results_dir = hyper_params.checkpoint_dir + args.tag + '/' + hyper_params.dataset + '/' + 'results_test_col.json'
        with open(results_dir, 'w') as json_file:
            json.dump(str(results), json_file)
        print(f"Scene: {hyper_params.dataset}", *[f"{meter}: {value:.8f}" for meter, value in results.items()])

