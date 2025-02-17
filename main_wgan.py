import os
import datetime
import warnings
from argparse import ArgumentParser
import pickle
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.feature_orderings import set_ordering
from constraints_code.parser import parse_constraints_file
from gather_results.reeval_final import prepare_gen_data
from utils import set_seed, read_csv, all_div_gt_n, _load_json
from data_processors.wgan.helpers import prepare_data_torch_scaling
from synthetizers.WGAN.wgan import train_model, sample, SingleTaskDataset
from evaluation.eval import eval_synthetic_data, sdv_eval_synthetic_data, constraints_sat_check

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)

from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat

import wandb

# wandb.log({'accuracy': train_acc, 'loss': train_loss})
# wandb.config.dropout = 0.2
# wandb.alert(title="Low accuracy", text=f"Accuracy {acc} is below threshold {thresh}")
# https://docs.wandb.ai/guides/data-and-model-versioning/dataset-versioning?_gl=1*1la1mgf*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MjguMTMuMC4w
# https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1?_gl=1*1p4t0h4*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MTUuMjYuMC4w
# https://docs.wandb.ai/guides/data-vis/tables-quickstart
DATETIME = datetime.datetime.now()

def get_args():
    args = ArgumentParser()
    args.add_argument("--seed", default=2, type=int)
    args.add_argument("--use_only_target_original_dtype", action='store_true')
    args.add_argument("--wandb_project", default='WGAN-fixed-rerun', type=str)
    args.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    args.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde', 'wasserstein', 'jsd', 'causal'])
    args.add_argument("--save_every_n_epochs", default=5, type=int)
    args.add_argument("--pac", default=1, type=int)
    args.add_argument("--epochs", default=400, type=int)
    args.add_argument('--optimiser', type=str, default="rmsprop", choices=['adam','rmsprop','sgd'], help='')
    args.add_argument("--batch_size", default=70, type=int)
    args.add_argument("--disc_repeats", default=2, type=int)
    args.add_argument("--gp_weight", default=10, type=float)
    args.add_argument("--d_lr", default=0.00005, type=float)
    args.add_argument("--g_lr", default=0.00005, type=float)
    args.add_argument("--alpha", default=0.9, type=float)
    args.add_argument("--weight_decay", default=0, type=float)
    args.add_argument("--momentum", default=0, type=float)  # 0.00005
    args.add_argument("--clamp", default=None, type=float)  # 0.01
    args.add_argument("--enable_sigmoid", default=False, action='store_true')
    args.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    args.add_argument("use_case", type=str, choices=["url","wids","botnet","lcld","heloc","news","faults",'kc','cc'])
    args.add_argument('--skip_evaluation', action='store_true')
    args.add_argument('--runtime_evaluation_only', action='store_true')
    return args.parse_args()


def postprocessing(scaler, sampled, gan):
    x = scaler.inverse_transform(sampled)
    cons_layer = correct_preds(x, gan.generator.ordering, gan.generator.sets_of_constr)
    check_all_constraints_sat(cons_layer, gan.generator.constraints)
    sampled = scaler.transform(cons_layer)
    return sampled

def sample_data(cast_types, sampled_data, gan, scaler, columns, roundable_idx, round_digits):
    if args.version == "postprocessing" or args.version == 'constrained':
        sampled_data = postprocessing(scaler, sampled_data, gan)
    sampled_data = sampled_data.detach().numpy()
    sampled_data = pd.DataFrame(sampled_data, columns=columns)
    sampled_data = sampled_data.astype(cast_types)
    
    #sampled_data[:,int_cols] = np.trunc(sampled_data[:,int_cols])
    #pd.DataFrame(sampled_data).to_csv(f"{path_name}/samples_{r}.csv")
    return sampled_data


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    ######################################################################
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################


    if args.use_case =="botnet":
        X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/tiny/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")

    else:
        X_train, (_, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
        X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")

    cast_types = X_train.dtypes
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes
    X_train_scaled, scaler = prepare_data_torch_scaling(X_train, args.use_case, cat_idx)

    train_ds = SingleTaskDataset(X_train_scaled)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    args.input_length = X_train_scaled.shape[1]
    args.bin_cols_idx = cat_idx

    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.disc_repeats}_{args.gp_weight}_{args.d_lr}_{args.g_lr}_{args.enable_sigmoid}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    path_name = f"outputs/WGAN_out/{args.use_case}/{args.version}/{exp_id}"
    args.exp_path = path_name

    # if not os.path.exists(path_name):
    # Create a new directory for the output
    os.makedirs(path_name)
    args.path_name = path_name


    # set args.pac:
    if args.pac != 1:
        if args.batch_size % args.pac != 0:
            original_pac = args.pac
            args.pac = all_div_gt_n(original_pac, args.batch_size)
            print(f'Changed pac {original_pac} to {args.pac}')

    ######################################################################
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True,  mode=args.wandb_mode)
    for k,v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################
    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################

    print("Created new directory {:}, the experiment is starting".format(path_name))
    gan = train_model(args, train_loader, scaler, path_name)

    ########################################################################################
    ################################  EVALUATION ###########################################
    ########################################################################################
    if args.use_case == "botnet" or args.use_case == "lcld":
        X_train = pd.read_csv(f"data/{args.use_case}/tiny/train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")

    args.sampling_sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]

    num_sampling_rounds = 5
    gen_data = [[], [], []]
    unconstrained_gen_data = [[], [], []]
    constrained_unrounded_gen_data = [[], [], []]
    sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]

    ordering, constraints = parse_constraints_file(args.constraints_file)
    ordering = set_ordering(args.use_case, ordering, args.label_ordering, 'wgan')
    ordering_list = ordering
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)


    if args.runtime_evaluation_only:
        size = 1000
        runs = []
        for i in range(num_sampling_rounds):
            start = timer()
            #sampled_data, unconstrained_output = sample(args, gan, size, X_train_scaled.shape[1], ordering_list, sets_of_constr)
            sampled_data = sample(args, gan, size, X_train_scaled.shape[1], ordering_list, sets_of_constr)
            end = timer()
            runtime = end - start
            runs.append(runtime)
        runtime_df = pd.DataFrame(list(zip([np.mean(runs)],[np.std(runs)])), columns=["Mean", "Std"])
        wandb.log({'Runtime/Sampling': runtime_df})

    else:

        gan.generator.eval()
        for r in range(num_sampling_rounds):
            for i in range(len(sizes)):
                constrained_unrounded_output, unconstrained_output = sample(args, gan, sizes[i], X_train_scaled.shape[1], ordering_list, sets_of_constr)
                unconstrained_gen_data[i].append(unconstrained_output)

                constrained_unrounded_output = pd.DataFrame(constrained_unrounded_output, columns=columns)
                constrained_unrounded_output = constrained_unrounded_output.astype(float)
                target_col = columns[-1]
                constrained_unrounded_output[target_col] = constrained_unrounded_output[target_col].astype(X_train.dtypes[-1])
                constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

                gen_data[i].append(constrained_unrounded_output)

        real_data = {"train": X_train, "val": X_val, "test": X_test}
        generated_data = {"train": gen_data[0], "val": gen_data[1], "test": gen_data[2]}
        unconstrained_generated_data = {"train": unconstrained_gen_data[0], "val": unconstrained_gen_data[1], "test": unconstrained_gen_data[2]}

        with open(f'{path_name}/generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        with open(f'{path_name}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)

        wandb.finish()

        ######################################################################
        args.real_data_partition = 'test'
        args.model_type = 'wgan'
        args.wandb_project = f"DRL_evaluation_{args.model_type}_{args.use_case}"

        wandb_run = wandb.init(project=args.wandb_project, id=exp_id)
        for k, v in args._get_kwargs():
            wandb_run.config[k] = v
        ######################################################################
        args.round_before_cons = False
        args.round_after_cons = False
        args.postprocessing = False
        if args.version != 'unconstrained':
            args.version = args.label_ordering

        generated_data, unrounded_generated_data = prepare_gen_data(args, unconstrained_generated_data, roundable_idx, round_digits, columns, X_train)

        # if args.seed < 3:
        constraints_sat_check(args, real_data, unrounded_generated_data, log_wandb=True)
        # sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
        #                         problem_type=dataset_info["problem_type"],
        #                         target_utility=dataset_info["target_col"], target_detection="", log_wandb=True,
        #                         wandb_run=wandb_run)
        print('Using evaluators with the following specs', dataset_info["problem_type"], dataset_info["target_size"],
            dataset_info["target_col"])
        eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
                            problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"],
                            target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True,
                            wandb_run=wandb_run, unrounded_generated_data_for_cons_sat=unrounded_generated_data)

