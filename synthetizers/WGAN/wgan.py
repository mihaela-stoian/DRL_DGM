import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, ReLU, Sequential
from torch.utils.data import Dataset

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering
from tqdm import tqdm

# from helpers.eval import  eval
from evaluation.constraints import constraint_satisfaction
from torch.nn import functional
from utils import round_func_BPDA

# from synthetizers.constrained_layer import get_constr_out
warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)
import wandb
import matplotlib.pyplot as plt


class SingleTaskDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_i = self.data[idx]
        return data_i


class Discriminator(nn.Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, pac):
        super(Discriminator, self).__init__()
        discriminator_dim = (256, 256)
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        return self.seq(input_.view(-1, self.pacdim))

    def gradient_penalty(self, real_data, fake_data, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1)
        alpha = alpha.repeat(1, self.pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_view = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_
        return gradient_penalty


class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):
    """Generator for the CTGAN."""

    def __init__(self, args, data_dim, enable_sigmoid, scaler, int_cols, bin_cols_idx):
        super(Generator, self).__init__()
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = self.get_sets_constraints(args.label_ordering,
                                                                                         args.constraints_file)
        self.version = args.version
        self.enable_sigmoid = enable_sigmoid
        self.scaler = scaler
        self.int_cols = int_cols
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.input_length = data_dim
        self.bin_cols_idx = bin_cols_idx
        generator_dim = (256, 256)
        dim = 128
        dim = data_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        x = data.clone()
        scaled = []
        scaled.append(torch.sigmoid(data[:, :len(self.scaler.num_idx)]))

        st = len(self.scaler.num_idx)
        for i, bin_col in enumerate(self.bin_cols_idx):
            end = st + self.scaler.ohe.categories_[i].shape[0]
            scaled.append(functional.gumbel_softmax(data[:, st:end], tau=0.2, hard=True))
            # scaled.append(round_func_BPDA(torch.softmax(data[:, st:end], dim=1)))
            st = end
        scaled = torch.cat(scaled, dim=1)

        if self.version == "constrained":
            if self.training:
                inverse = self.scaler.inverse_transform(scaled)
                # cons_layer = get_constr_out(x)
                cons_layer = correct_preds(inverse, self.ordering, self.sets_of_constr)

                # check_all_constraints_sat(cons_layer, self.constraints)

                output_cons = self.scaler.transform(cons_layer)
                # output_cons = self.scaler.transform(inverse)
                # for i in range(output_cons.shape[1]):
                #     np.testing.assert_almost_equal(scaled[:,i].detach().numpy() , output_cons[:,i].detach().numpy() )
                # data_out = torch.concat([output_cons[:, :len(self.scaler.num_idx)], scaled[:, len(self.scaler.num_idx):]], dim=1)

                return output_cons
            else:
                return scaled
        else:
            return scaled

    def get_sets_constraints(self, label_ordering_choice, constraints_file):
        ordering, constraints = parse_constraints_file(constraints_file)
        # set ordering
        ordering = set_ordering(self.args.use_case, ordering, label_ordering_choice, 'wgan')

        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        return constraints, sets_of_constr, ordering


class WGAN():
    def __init__(self, args, generator, discriminator, train_data, test_data, scaler):
        self.args = args
        self.discriminator = discriminator
        self.generator = generator
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        self.int_cols = args.int_col
        self.use_case = args.use_case
        self.clamp = args.clamp
        self.optimiser = args.optimiser

        if self.optimiser == 'rmsprop':
            self.discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr,
                                                               alpha=args.alpha, momentum=args.momentum,
                                                               weight_decay=args.weight_decay)
            self.generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr, alpha=args.alpha,
                                                           momentum=args.momentum, weight_decay=args.weight_decay)
        elif self.optimiser == 'adam':
            self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(
            0.5, 0.9))  # used betas as recommended by ctgan, which uses adam
            self.generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
        elif self.optimiser == 'sgd':
            self.discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.d_lr,
                                                           momentum=args.momentum, weight_decay=args.weight_decay)
            self.generator_optimizer = torch.optim.SGD(generator.parameters(), lr=args.g_lr, momentum=args.momentum,
                                                       weight_decay=args.weight_decay)
        else:
            pass
        self.discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.d_lr, alpha=args.alpha,
                                                           momentum=args.momentum, weight_decay=args.weight_decay)
        self.generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.g_lr, alpha=args.alpha,
                                                       momentum=args.momentum, weight_decay=args.weight_decay)

    def train_discriminator(self, true_data, generated_data, gp_weight):
        with torch.autograd.set_detect_anomaly(True):
            ## Train Discriminator in real and synthetic data
            self.discriminator_optimizer.zero_grad()
            d_real_loss = torch.mean(self.discriminator(true_data))
            d_syn_loss = torch.mean(self.discriminator(generated_data.detach()))
            gp = self.discriminator.gradient_penalty(true_data, generated_data, gp_weight)
            gp.backward(retain_graph=True)
            discriminator_loss = d_syn_loss - d_real_loss
            discriminator_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip)
            self.discriminator_optimizer.step()
        return d_real_loss, d_syn_loss, discriminator_loss

    def eval_cons_layer(self, input_length):
        self.generator.eval()
        noise = torch.rand(size=(self.test_data.shape[0], input_length)).float()
        with torch.no_grad():
            generated_data = self.generator(noise)
        generated_data = self.scaler.inverse_transform(generated_data)
        features = generated_data[:, :-1].detach().numpy()
        cons_rate, batch_rate, ind_score = constraint_satisfaction(features, "url")
        self.generator.train()
        return cons_rate, batch_rate, ind_score

    def train_generator(self, shape):
        with torch.autograd.set_detect_anomaly(True):
            self.generator_optimizer.zero_grad()
            noise = torch.rand(size=(shape[0], shape[1])).float()
            generated_data = self.generator(noise)
            discriminator_out = self.discriminator(generated_data)
            generator_loss = -torch.mean(
                discriminator_out)  # + torch.nn.functional.l1_loss(generated_data, generated_data[:, torch.randperm(generated_data.shape[1])])
            generator_loss.backward()
            self.generator_optimizer.step()
        return generator_loss, generated_data

    def train_step(self, true_data, disc_repeats=1, gp_weight=1):

        mean_d, mean_d_syn, mean_d_real = 0, 0, 0
        for i in range(disc_repeats):
            # clamp parameters to a cube # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
            if self.clamp is not None:
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)

            noise = torch.rand(size=(true_data.shape[0], true_data.shape[1])).float()
            generated_data = self.generator(noise)
            d_real_loss, d_syn_loss, discriminator_loss = self.train_discriminator(true_data, generated_data, gp_weight)
            mean_d_syn += d_syn_loss
            mean_d_real += d_real_loss
            mean_d += discriminator_loss
            # wandb.log({'steps/1step_disc_real': d_real_loss, 'steps/1step_disc_syn': d_syn_loss, 'steps/1step_disc': discriminator_loss})
        generator_loss, syn_data = self.train_generator((true_data.shape[0], true_data.shape[1]))

        loss_d_syn = mean_d_syn / disc_repeats
        loss_d_real = mean_d_real / disc_repeats
        loss_d = mean_d / disc_repeats
        # wandb.log({'steps/gen_loss': generator_loss, 'steps/disc_loss': loss_d})
        return generator_loss.item(), loss_d_syn.item(), loss_d_real.item(), loss_d.item()


def train_model(args, train_loader, scaler, path_name):
    # print('Starting experiment with constraints')

    if args.use_case == "botnet" or args.use_case == "lcld":
        train_data = pd.read_csv(f"data/{args.use_case}/tiny/train_data.csv")
        test_data = pd.read_csv(f"data/{args.use_case}/tiny/test_data.csv")
    else:
        train_data = pd.read_csv(f"data/{args.use_case}/train_data.csv")
        test_data = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    int_col_names = train_data.select_dtypes(include=np.number).columns.tolist()
    int_col = [train_data.columns.get_loc(c) for c in int_col_names]
    args.int_col = int_col

    # Models
    generator = Generator(args, args.input_length, args.enable_sigmoid, scaler, args.int_col, args.bin_cols_idx)
    discriminator = Discriminator(args.input_length, args.pac)
    gan = WGAN(args, generator, discriminator, train_data, test_data, scaler)

    # loss
    # loss_g_all, loss_d_syn_all,  loss_d_real_all, loss_d_all = [], [], [], []
    for epoch in range(args.epochs):
        loss_g_running, loss_d_syn_running, loss_d_real_running, loss_d_running = 0, 0, 0, 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # if data.shape[0]!=70:
            #     pass
            # else:
            real_data = data.float()
            generator_loss, loss_d_syn, loss_d_real, loss_d = gan.train_step(real_data, args.disc_repeats,
                                                                             args.gp_weight)
            loss_g_running += generator_loss
            loss_d_syn_running += loss_d_syn
            loss_d_real_running += loss_d_real
            loss_d_running += loss_d

        wandb.log({'epochs/epoch': epoch, 'epochs/loss_gen': loss_g_running / len(train_loader),
                   'epochs/loss_disc_syn': loss_d_syn_running / len(train_loader),
                   'epochs/loss_disc_real': loss_d_real_running / len(train_loader),
                   'epochs/loss_disc': loss_d_running / len(train_loader)})
        print('Epoch {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch,
                                                                                 loss_d_running / len(train_loader),
                                                                                 loss_g_running / len(train_loader)))
        print("Discriminator real {}, fake {}".format(loss_d_real_running / len(train_loader),
                                                      loss_d_syn_running / len(train_loader)))

        # cons_rate, batch_rate, ind_score  = gan.eval_cons_layer(data.shape[1])
        # wandb.log({'constraints/mean_ind_score': ind_score.mean(), 'constraints/batch_rate': batch_rate, 'constraints/cons_rate': cons_rate})
        # wandb.log({f'constraints/ind_score_{epoch}': ind_score[epoch] for epoch in range(len(ind_score))})

        # if args.use_case not in ['lcld', 'botnet']:
        #     if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
        #         torch.save(gan.generator, f"{path_name}/model_{epoch}.pt")
        # else:
        #     if epoch >= 5 and epoch % args.save_every_n_epochs == 0:
        #         torch.save(gan.generator, f"{path_name}/model_{epoch}.pt")

    PATH = f"{path_name}/model.pt"
    torch.save(gan.generator, PATH)

    return gan


def sample(args, gan, n, input_length, ordering_list, sets_of_constr):
    """Sample data similar to the training data

    Args:
        n (int):
            Number of rows to sample.
    Returns:
        numpy.ndarray or pandas.DataFrame
    """
    gan.generator.eval()
    # torch.manual_seed(1234)
    noise = torch.rand(size=(n, input_length)).float()
    with torch.no_grad():
        generated_data = gan.generator(
            noise)  # it always returns the unconstrained, scaled (so before inverse is applied, even if version was constrained)
    generated_data = gan.generator.scaler.inverse_transform(generated_data)
    unconstrained_data = generated_data.clone().detach()
    if args.version == "postprocessing" or args.version == 'constrained':
        generated_data = correct_preds(generated_data, ordering_list, sets_of_constr)

    sampled_data = generated_data.detach().numpy()
    # sampled_data = pd.DataFrame(sampled_data, columns=args.train_data_cols)
    # sampled_data = sampled_data.astype(args.dtypes)

    return sampled_data, unconstrained_data
