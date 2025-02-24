"""CLI."""
import os
import argparse
import pickle
import datetime
import pandas as pd
import numpy as np
import wandb

from timeit import default_timer as timer

from synthetizers.CTGAN.ctgan import CTGAN
from evaluation.eval import eval_synthetic_data, sdv_eval_synthetic_data, constraints_sat_check
from utils import set_seed, read_csv, all_div_gt_n, _load_json
from gather_results.reeval_final import prepare_gen_data

# wandb.log({'accuracy': train_acc, 'loss': train_loss})
# wandb.config.dropout = 0.2
# wandb.alert(title="Low accuracy", text=f"Accuracy {acc} is below threshold {thresh}")
# https://docs.wandb.ai/guides/data-and-model-versioning/dataset-versioning?_gl=1*1la1mgf*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MjguMTMuMC4w
# https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1?_gl=1*1p4t0h4*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MTUuMjYuMC4w
# https://docs.wandb.ai/guides/data-vis/tables-quickstart
DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--use_only_target_original_dtype", action='store_true')
    parser.add_argument("--pac", default=10, type=int)
    parser.add_argument("--wandb_project", default="ctgan", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument("--save_every_n_epochs", default=5, type=int)
    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde', 'wasserstein', 'jsd', 'causal'])

    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')

    parser.add_argument('--sample_condition_column', default=None, type=str,
                        help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str,
                        help='Specify the value of the selected discrete column.')
    parser.add_argument("use_case", type=str, choices=["url","wids","botnet","lcld","heloc","news","faults",'kc','cc'])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--runtime_evaluation_only', action='store_true')

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    set_seed(args.seed)
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.discriminator_lr}_{args.generator_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    path = f"outputs/CTGAN_out/{args.use_case}/{args.version}/{exp_id}"
    args.exp_path = path
    os.makedirs(path)


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
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################
    if args.use_case == "botnet":
        X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/tiny/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")

    else:
        X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(f"data/{args.use_case}/train_data.csv", args.use_case, dataset_info["manual_inspection_categorical_cols_idx"])
        X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/val_data.csv")
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    if cat_cols == None:
        cat_cols = []
        cat_idx = []


    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(X_test,
            embedding_dim=args.embedding_dim, generator_dim=generator_dim,
            discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
            generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
            epochs=args.epochs, path=path, bin_cols_idx=cat_idx, version=args.version, pac=args.pac,
                      feats_in_constraints=dataset_info["feats_in_constraints"])

    model.set_random_state(args.seed)
    model.fit(args, X_train, cat_cols)

    # args.save = f'{path}/final_ctgan_model.pt'
    if args.save is not None:
        model.save(args.save)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    if args.use_case == "botnet" or args.use_case == "lcld":
        X_train = pd.read_csv(f"data/{args.use_case}/tiny/train_data.csv")
        X_test = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
        X_val = pd.read_csv(f"data/{args.use_case}/tiny/val_data.csv")
    args.sampling_sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        
    model.set_random_state(args.seed)
    num_sampling_rounds = 5

    if args.runtime_evaluation_only:
        size = 1000
        runs = []
        for i in range(num_sampling_rounds):
            start = timer()
            sampled_data, unconstrained_output = model.sample(size, args.sample_condition_column, args.sample_condition_column_value)
            end = timer()
            runtime = end - start
            runs.append(runtime)
        runtime_df = pd.DataFrame(list(zip([np.mean(runs)],[np.std(runs)])), columns=["Mean", "Std"])
        wandb.log({'Runtime/Sampling': runtime_df})
    
    else:

        gen_data = [[], [], []]
        unconstrained_gen_data = [[], [], []]
        constrained_unrounded_gen_data = [[], [], []]
        sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        for r in range(num_sampling_rounds): 
            for i in range (len(sizes)):
                sampled_data, unconstrained_output = model.sample(sizes[i], args.sample_condition_column, args.sample_condition_column_value)
                unconstrained_gen_data[i].append(unconstrained_output)
                constrained_unrounded_output = sampled_data
                constrained_unrounded_output = pd.DataFrame(constrained_unrounded_output, columns=columns)
                constrained_unrounded_output = constrained_unrounded_output.astype(float)
                target_col = columns[-1]
                constrained_unrounded_output[target_col] = constrained_unrounded_output[target_col].astype(X_train.dtypes[-1])
                constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

                # sampled_data = pd.DataFrame(sampled_data, columns=columns)
                # sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)  # NOTE: this shouldn't be after the constraints have been applied! (fixed by removing constr correction from sample fc, and adding it below here)
                # sampled_data = sampled_data.astype(X_train.dtypes)

                gen_data[i].append(constrained_unrounded_output)


        generated_data = {"train":gen_data[0], "val":gen_data[1], "test":gen_data[2]}
        unconstrained_generated_data = {"train":unconstrained_gen_data[0], "val":unconstrained_gen_data[1], "test":unconstrained_gen_data[2]}
        constrained_unrounded_generated_data = {"train":constrained_unrounded_gen_data[0], "val":constrained_unrounded_gen_data[1], "test":constrained_unrounded_gen_data[2]}

        with open(f'{path}/generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        with open(f'{path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{path}/constrained_unrounded_generated_data.pkl', 'wb') as f:
            pickle.dump(constrained_unrounded_generated_data, f)

        real_data = {"train": X_train, "val": X_val, "test": X_test}



        if not args.skip_evaluation: 

            wandb.finish()
            ######################################################################
            args.real_data_partition = 'test'
            args.model_type = 'ctgan'

            if 'hyerparam' in args.wandb_project or 'hyper' in args.wandb_project:
                args.wandb_project = f"DRL_evaluation_{args.model_type}_{args.use_case}_hyperparam_search"
            else:
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


            # if args.seed < 3:
            #     constraints_sat_check(args, real_data, generated_data, log_wandb=True)
            #     sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_detection="", log_wandb=True, wandb_run=wandb_run)
            #     eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True, wandb_run=wandb_run)

if __name__ == '__main__':

    main()
