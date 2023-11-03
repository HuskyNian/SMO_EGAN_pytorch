from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from utils.metric import *

import warnings
warnings.filterwarnings('ignore')
from numpy.lib.function_base import append
from ctgan import CTGAN
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy

import numpy as np
import pandas as pd
import os
import random
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

from collections import namedtuple

from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder
def set_seed(seed=42):
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_diagrams(tcaps,utilities,g_losses,cond_losses,d_losses,index,census='UK'):
    plt.plot(np.arange(len(g_losses)),g_losses,label='generator loss')
    plt.plot(np.arange(len(g_losses)),cond_losses,label='generator condition loss')
    plt.xlabel('training epochs')
    plt.legend()
    plt.savefig(os.path.join(f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/imgs',f'hist_g_loss_{index}.png'))
    plt.close()
    #plt.show()
    plt.plot(np.arange(len(g_losses)),d_losses,label='discriminator loss')
    plt.xlabel('training epochs')
    plt.legend()
    plt.savefig(os.path.join(f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/imgs',f'hist_d_loss_{index}.png'))
    plt.close()
    #plt.show()
    plt.plot(np.arange(len(g_losses)),tcaps,label='risk')
    plt.plot(np.arange(len(g_losses)),utilities,label='utility')
    plt.xlabel('training epochs')
    plt.legend()
    plt.savefig(os.path.join(f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/imgs',f'hist_risk_utility_{index}.png'))
    plt.close()

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)

class my_DataTransformer(DataTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(model_missing_values=False, max_clusters=min(len(data), 10))
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
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


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


def random_state(function):
    """Set the random state before calling the function.
    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper
import torch.nn.functional as F
myfake = None
class my_ctgan(CTGAN):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def make_ordinal_encoder(self,orig):
        key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
        # get columns used and make y to be binary
        orig = orig[key_cols]
        orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
        orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
        encoder = OrdinalEncoder()
        encoder.fit(orig)
        return encoder

    def fit(self, train_data, discrete_columns=(), epochs=None,index=1,census='UK'):
        """Fit the CTGAN Synthesizer models to the training data.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        if census=='UK':
            self.ordinal_encoder = self.make_ordinal_encoder(train_data)
        else:
            self.ordinal_encoder = None
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = my_DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        orig_data = train_data.copy()
        print('start transform data')
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        #print('discriminator:',data_dim + self._data_sampler.dim_cond_vec(),self._discriminator_dim, self.pac)
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)
        #print('generator:',self._embedding_dim + self._data_sampler.dim_cond_vec(),self._generator_dim, data_dim)
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        best_utility = 0.
        best_tcap = 0.
        tcaps = []
        utilities =[]
        g_losses = []
        cond_losses = []
        d_losses = []
        print('starting training')
        with open(f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/results_{index}.csv','w',encoding='utf-8') as f:
            f.write('tcap,cio,uni_roc,bi_roc,utility\n')
        start = time.time()
        print_time = True
        for i in range(epochs):

            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                #print('g optimizer 1 times')
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                #loss_g = -F.binary_cross_entropy(torch.nn.functional.sigmoid(y_fake), torch.zeros_like(y_fake)) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            if self._verbose:
                xfake = self.sample(len(orig_data))
                #xfake = self.sample(6600)
                global myfake
                myfake = xfake
                tcap = cal_mean_tcap(census,orig_data,xfake)
                cio = cal_mean_cio(census,orig_data,xfake,self.ordinal_encoder)
                uni_roc,bi_roc = cal_mean_roc(census,orig_data,xfake)
                utility = (uni_roc+bi_roc+cio)/3

                #plot diagrams
                tcaps.append(tcap)
                utilities.append(utility)
                cond_losses.append(cross_entropy.cpu().detach().numpy())
                g_losses.append(loss_g.cpu().detach().numpy()-cross_entropy.cpu().detach().numpy())
                d_losses.append(loss_d.cpu().detach().numpy())
                save_diagrams(tcaps,utilities,g_losses,cond_losses,d_losses,index,census=census)

                torch.save(self._generator.state_dict(),f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/last_ctgan_{index}.pth')
                is_best = (utility-best_utility)*2 + max(best_tcap,0) - max(tcap,0)
                #if utility>best_utility:
                if is_best>=0:
                    best_utility = utility
                    best_tcap = tcap
                    torch.save(self._generator.state_dict(),f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/best_ctgan_{index}.pth')
                with open(f'/content/drive/MyDrive/Aresearch/SMO_EGAN_syn/ctgan_{census}/results_{index}.csv','a',encoding='utf-8') as f:
                    f.write(f'{tcap},{cio},{uni_roc},{bi_roc},{utility}\n')
                print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      f'tcap: {tcap: .4f}',f'utility: {utility: .4f}',f'cio: {cio: .4f}',f'uni_roc: {uni_roc: .4f}',f'bi_roc: {bi_roc: .4f}',
                      flush=True)
                print('best utility and tcap:',best_utility,best_tcap)
                if print_time:
                    print_time=False
                    print('one epoch time:',time.time()-start)
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["UK","Rwanda","Fiji","Canada"], default="UK")
    parser.add_argument("--suffix", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problem = args.problem 
    set_seed(args.seed)
    if problem=='UK':
        df = pd.read_spss('/content/drive/MyDrive/Aresearch/UKDA-7210-spss/spss/gb91ind.sav')
        df = df.loc[df['REGIONP']=='West Midlands']
        df = df[['AREAP','AGE','COBIRTH','ECONPRIM','ETHGROUP','FAMTYPE','HOURS','LTILL','MSTATUS','QUALNUM',
         'RELAT','SEX','SOCLASS','TRANWORK','TENURE']]
        cate_cols = []
        for col in df.columns:
            if df[col].dtype=='category':
                cate_cols.append(col)
                df[col] = df[col].astype('str')
        print(f'there are {len(cate_cols)} category cols')
        #assert nPassD==1 ############
    elif problem=='Canada':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Canada2011_census_microdata.csv')
        df.columns = [i.replace('CA2011A_','') for i in df.columns] 
        cate_cols = []
        c_cols = ['HRSWK','INCTOT','WKSWORK']
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')
    elif problem=='Fiji':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Fiji2007_census_microdata.csv')
        df.columns = [i.replace('FJ2007A_','') for i in df.columns] 
        cate_cols = []
        c_cols = []
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')
    elif problem=='Rwanda':
        df = pd.read_csv('/content/drive/MyDrive/Aresearch/census/Rwanda2012_census_microdata.csv')
        df.columns = [i.replace('RW2012A_','') for i in df.columns] 
        cate_cols = []
        c_cols = []
        for col in df.columns:
            if col not in c_cols:
                cate_cols.append(col)
        print(f'there are {len(cate_cols)} category cols')
    ctgan = my_ctgan(epochs=args.epochs,verbose=args.verbose)
    ctgan.fit(df,cate_cols,index=args.suffix,census=problem)
