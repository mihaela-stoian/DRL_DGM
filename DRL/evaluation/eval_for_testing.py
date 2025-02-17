import torch
import wandb
import pandas as pd
import numpy as np

from constraints_code.parser import parse_constraints_file

def constraints_sat_check(args, real_data, generated_data, log_wandb):
    # Note: the ordering of the labels does not matter here
    _, constraints = parse_constraints_file(args.constraints_file)
    gen_sat_check(args, generated_data, constraints, log_wandb)
    real_sat_check(args, real_data, constraints, log_wandb)

def gen_sat_check(args, generated_data, constraints, log_wandb):
    sat_rate_per_constr = {i:[] for i in range(len(constraints))}
    percentage_cons_sat_per_pred = []
    percentage_of_samples_sat_constraints = []
    percentage_of_constr_violated_at_least_once = []

    for _, gen_data in enumerate(generated_data["train"]):
        samples_sat_constr = torch.ones(gen_data.shape[0]) == 1.
        num_cons_sat_per_pred = torch.zeros(gen_data.shape[0])
        num_constr_violated_at_least_once = 0.
        # gen_data = gen_data.iloc[:, :-1].to_numpy()
        gen_data = torch.tensor(gen_data.to_numpy())
        for j, constr in enumerate(constraints):
            sat_per_datapoint = constr.disjunctive_inequality.check_satisfaction(gen_data)
            num_cons_sat_per_pred += sat_per_datapoint*1.
            num_constr_violated_at_least_once += 0. if sat_per_datapoint.all() else 1.
            sat_rate = sat_per_datapoint.sum()/len(sat_per_datapoint)
            # print('Synth sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
            sat_rate_per_constr[j].append(sat_rate)
            samples_sat_constr = samples_sat_constr & sat_per_datapoint
            # print('samples_violating_constr:', samples_violating_constr.sum())
        percentage_cons_sat_per_pred.append(np.array(num_cons_sat_per_pred/len(constraints)).mean())
        percentage_of_samples_sat_constraints.append(sum(samples_sat_constr) / len(samples_sat_constr))
        percentage_of_constr_violated_at_least_once.append(num_constr_violated_at_least_once/len(constraints))
    sat_rate_per_constr = {i:[sum(sat_rate_per_constr[i])/len(sat_rate_per_constr[i]) * 100.0] for i in range(len(constraints))}
    percentage_cons_violations_per_pred = 100.0-sum(percentage_cons_sat_per_pred)/len(percentage_cons_sat_per_pred) * 100.0
    percentage_of_samples_violating_constraints = 100.0-sum(percentage_of_samples_sat_constraints)/len(percentage_of_samples_sat_constraints) * 100.0
    percentage_of_constr_violated_at_least_once = sum(percentage_of_constr_violated_at_least_once)/len(percentage_of_constr_violated_at_least_once) * 100.0
    print('SYNTH', 'sat_rate_per_constr', sat_rate_per_constr)
    print('SYNTH', 'percentage_of_samples_violating_at_least_one_constraint', percentage_of_samples_violating_constraints)
    print('SYNTH', 'percentage_cons_violations_per_pred', percentage_cons_violations_per_pred)
    print('SYNTH', 'percentage_of_constr_violated_at_least_once', percentage_of_constr_violated_at_least_once)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    if log_wandb:
        wandb.log({f"INFERENCE/synth_constr_sat": wandb.Table(dataframe=sat_rate_per_constr)})

    synth_constr_eval_metrics = pd.DataFrame({'percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints],
                                              'percentage_cons_violations_per_pred': percentage_cons_violations_per_pred,
                                              'percentage_of_constr_violated_at_least_once': percentage_of_constr_violated_at_least_once},
                                             columns=['percentage_of_samples_violating_constraints', 'percentage_cons_violations_per_pred', 'percentage_of_constr_violated_at_least_once'])
    if log_wandb:
        wandb.log({f"INFERENCE/synth_constr_eval_metrics": wandb.Table(dataframe=synth_constr_eval_metrics)})

    return sat_rate_per_constr, percentage_of_samples_violating_constraints, synth_constr_eval_metrics


def real_sat_check(args, real_data, constraints, log_wandb):
    sat_rate_per_constr = {i: [] for i in range(len(constraints))}
    percentage_of_samples_sat_constraints = []

    real_data = real_data["train"]
    samples_sat_constr = torch.ones(real_data.shape[0]) == 1.
    # real_data = real_data.iloc[:, :-1].to_numpy()
    real_data = torch.tensor(real_data.to_numpy())

    for j, constr in enumerate(constraints):
        sat_per_datapoint = constr.disjunctive_inequality.check_satisfaction(real_data)
        sat_rate = sat_per_datapoint.sum() / len(sat_per_datapoint)
        # print('Real sat_rate is', sat_rate, sat_per_datapoint.sum(), len(sat_per_datapoint), sat_per_datapoint)
        sat_rate_per_constr[j].append(sat_rate)
        samples_sat_constr = samples_sat_constr & sat_per_datapoint

    percentage_of_samples_sat_constraints.append(sum(samples_sat_constr)/len(samples_sat_constr))
    sat_rate_per_constr = {i: [sum(sat_rate_per_constr[i]) / len(sat_rate_per_constr[i]) * 100.0] for i in
                           range(len(constraints))}
    percentage_of_samples_violating_constraints = 100.0 - sum(percentage_of_samples_sat_constraints) / len(
        percentage_of_samples_sat_constraints) * 100.0
    print('REAL', 'sat_rate_per_constr', sat_rate_per_constr)
    print('REAL', 'percentage_of_samples_violating_constraints', percentage_of_samples_violating_constraints)

    sat_rate_per_constr = pd.DataFrame(sat_rate_per_constr, columns=list(range(len(constraints))))
    if log_wandb:
        wandb.log({f"INFERENCE/real_constr_sat": wandb.Table(dataframe=sat_rate_per_constr)})

    percentage_of_samples_violating_constraints = pd.DataFrame({'real_percentage_of_samples_violating_constraints': [percentage_of_samples_violating_constraints]}, columns=['real_percentage_of_samples_violating_constraints'])
    if log_wandb:
        wandb.log({f"INFERENCE/real_percentage_of_samples_violating_constr": wandb.Table(
        dataframe=percentage_of_samples_violating_constraints)})

    return sat_rate_per_constr, percentage_of_samples_violating_constraints
