from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds
from constraints_code.feature_orderings import set_ordering
from constraints_code.parser import parse_constraints_file
import pandas as pd
import pickle as pkl
import torch

from evaluation.eval_for_testing import gen_sat_check, real_sat_check



def constrain_and_check_constr_for_model(input_path, constraints_file, use_case, model_type, predefined_ordering, constraints_file_to_constrain_data=None, predef_ord=None):
    if constraints_file_to_constrain_data is None:
        constraints_file_to_constrain_data = constraints_file
    ordering, constraints = parse_constraints_file(constraints_file_to_constrain_data)
    if predefined_ordering == 'random':
        ordering = set_ordering(use_case, ordering, 'random', model_type)
    if predefined_ordering == 'predefined':
        if predef_ord is None:
            predef_ord = ordering
        ordering = set_ordering(use_case, predef_ord, 'predefined', model_type)
    else:
        ordering = set_ordering(use_case, ordering, predefined_ordering, model_type)
    sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)

    try:
        X_train = pd.read_csv(f"../data/{use_case}/val_data.csv")
    except:
        X_train = pd.read_csv(f"./data/{use_case}/val_data.csv")
    generated_data = pkl.load(open(f'{input_path}', 'rb'))
    if model_type.lower() == 'tablegan':
        try:
            generated_data = {"train": generated_data[0], "val": generated_data[1], "test": generated_data[2]}
        except:
            pass
    gen_data = {'train': [], 'val': [], 'test': []}
    for part in gen_data.keys():
        print("Part", part)
        for j in range(len(generated_data[part])):
            sampled_data = generated_data[part][j]

            from pandas import DataFrame
            if type(sampled_data) == DataFrame:
                sampled_data = sampled_data.to_numpy()
            sampled_data = torch.tensor(sampled_data)
            sampled_data = correct_preds(sampled_data, ordering, sets_of_constr)
            # sat = check_all_constraints_sat(sampled_data, constraints, error_raise=False)

            if isinstance(sampled_data, torch.Tensor):
                sampled_data =  sampled_data.detach().numpy()
            sampled_data = pd.DataFrame(sampled_data, columns=X_train.columns)
            # sampled_data = sampled_data.astype(X_train.dtypes)
            gen_data[part].append(sampled_data)

    _, constraints = parse_constraints_file(constraints_file)
    sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics = gen_sat_check(None,
                                                                                                                gen_data,
                                                                                                                constraints,
                                                                                                                log_wandb=False)
    return sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics, gen_data



def check_constr_for_model(input_path, constraints_file, use_case, model_type):
    X_train = pd.read_csv(f"../data/{use_case}/val_data.csv")
    generated_data = pkl.load(open(f'{input_path}', 'rb'))
    if model_type.lower() == 'tablegan':
        try:
            generated_data = {"train": generated_data[0], "val": generated_data[1], "test": generated_data[2]}
        except:
            pass
    gen_data = {'train': [], 'val': [], 'test': []}
    for part in gen_data.keys():
        print("Part", part)
        for j in range(len(generated_data[part])):
            sampled_data = generated_data[part][j]
            if isinstance(sampled_data, torch.Tensor):
                sampled_data =  sampled_data.detach().numpy()
            sampled_data = pd.DataFrame(sampled_data, columns=X_train.columns)
            # sampled_data = sampled_data.astype(X_train.dtypes)
            gen_data[part].append(sampled_data)
    _, constraints = parse_constraints_file(constraints_file)
    sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics = gen_sat_check(None,
                                                                                                                gen_data,
                                                                                                                constraints,
                                                                                                                log_wandb=False)
    return sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics


def check_constr_for_real_data(input_path, constraints_file):
    X_train = pd.read_csv(input_path)
    real_data = {"train": X_train}

    _, constraints = parse_constraints_file(constraints_file)
    sat_rate_per_constr, percentage_of_samples_violating_constraints = real_sat_check(None, real_data, constraints, log_wandb=False)
    return sat_rate_per_constr, percentage_of_samples_violating_constraints['real_percentage_of_samples_violating_constraints'][0]

