import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils.metric import *
from ctgan_model import my_ctgan,set_seed
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def plot(samples,df,census='UK',encoder=None):
    tcaps = []
    cios = []
    uni_rocs = []
    bi_rocs = []
    utilities = []
    pace = 500
    steps =int(np.ceil( len(samples) // 500))
    for step in range(steps):
        if step == steps-1:
            section = samples
            x = len(samples)
        else:
            section = samples[:pace*(step+1)]
            x = pace*(step+1)
        tcap = cal_mean_tcap(census,df,section)
        tcaps.append(tcap)
        cio = cal_mean_cio(census,df,section,encoder)
        cios.append(cio)
        uni_roc,bi_roc = cal_mean_roc(census,df,section)
        uni_rocs.append(uni_roc)
        bi_rocs.append(bi_roc)
        utilities.append((bi_roc+uni_roc+cio)/3)
        plt.figure(figsize=(17,10))
        plt.plot((np.arange(len(tcaps))+1)*pace,tcaps,label='tcap')
        plt.plot((np.arange(len(tcaps))+1)*pace,cios,label='cio')
        plt.plot((np.arange(len(tcaps))+1)*pace,uni_rocs,label='uni_roc')
        plt.plot((np.arange(len(tcaps))+1)*pace,bi_rocs,label='bi_roc')
        plt.plot((np.arange(len(tcaps))+1)*pace,utilities,label='utility')
        plt.legend()
        clear_output()
        plt.show()
    return tcaps,cios,uni_rocs,bi_rocs,utilities
def evaluate(ctgan,df,samples = 104267,path = None,census='UK',):
    if path is not None:
        ctgan._generator.load_state_dict(torch.load(path))
    ctgan._generator.eval()
    section = ctgan.sample(samples)
    tcap = cal_mean_tcap(census,df,section)
    cio = cal_mean_cio(census,df,section,ctgan.ordinal_encoder)
    uni_roc,bi_roc = cal_mean_roc(census,df,section)
    print(f'tcap:{tcap},cio:{cio},uni_roc{uni_roc},bi_roc{bi_roc},utility:{(bi_roc+uni_roc+cio)/3}')
    return tcap,cio,uni_roc,bi_roc,(bi_roc+uni_roc+cio)/3
def evaluate_n(ctgan,df,path=None,samples=None,n=5,census='UK'):
    results = []
    if samples is None:
        samples = len(df)
    print(f'samples{samples}')
    for i in range(n):
        set_seed(42+i)
        results.append(np.expand_dims( np.array(evaluate(ctgan,df,samples=samples,path=path,census=census)),axis=0)  )
    results = np.concatenate(results,axis=0)
    print('mean tcap,cio,uni_roc,bi_roc,utility',results.mean(axis=0))
import argparse
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["UK","Rwanda","Fiji","Canada"], default="UK")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate_times", type=int, default=1)
    parser.add_argument("--paths", nargs="+" )
    args = parser.parse_args()
    problem = args.problem
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
    ctgan = my_ctgan(epochs=1,verbose=True)
    ctgan.fit(df,cate_cols,index=0,census=args.problem)

    for p in args.paths:
        evaluate_n(ctgan,df,path=p,n=args.evaluate_times,census=args.problem)
    
