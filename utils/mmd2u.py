
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
from tqdm import tqdm
from dataset import datasets
from sklearn.metrics import pairwise_kernels, pairwise_distances

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()

def compute_metric_mmd2(X,Y, sigma2 = None):
    m = len(X)
    n = len(Y)
    if sigma2 is None:
        sigma2 = np.median(pairwise_distances(X, Y, metric='euclidean'))**2
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric='rbf',gamma=1.0/sigma2)
    mmd2u = MMD2u(K, m, n)
    return mmd2u

def quadratic_time_mmd(X,Y,kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)
       
    n = len(K_XX)
    m = len(K_YY)
    
    # IMPLEMENT: unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def gauss_kernel(X, Y=None, sigma=1.0):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X
    
    returns: kernel matrix
    """

    # IMPLEMENT: compute squared distances and kernel matrix
    sq_dists = sq_distances(X,Y)
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def sq_distances(X,Y=None):
    from scipy.spatial.distance import squareform, pdist, cdist
    assert(X.ndim==2)
    # IMPLEMENT: compute pairwise distance matrix. Don't use explicit loops, but the above scipy functions
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists

def two_sample_permutation_test(test_statistic, X, Y, num_permutations, prog_bar=True):
    assert X.ndim == Y.ndim
    statistics = np.zeros(num_permutations)
    range_ = range(num_permutations)
    if prog_bar:
        range_ = tqdm(range_)
    for i in range_:
        # concatenate samples
        if X.ndim == 1:
            Z = np.hstack((X,Y))
        elif X.ndim == 2:
            Z = np.vstack((X,Y))
            
        # IMPLEMENT: permute samples and compute test statistic
        perm_inds = np.random.permutation(len(Z))
        Z = Z[perm_inds]
        X_ = Z[:len(X)]
        Y_ = Z[len(X):]
        my_test_statistic = test_statistic(X_, Y_)
        statistics[i] = my_test_statistic
    return statistics

def mmd_toy_metric(X, dataname, samples = 1000, perms = 100, prog_bar = True):
    if type(dataname) is str:
        Y = datasets.toy_dataset(dataname, size=samples)
        sigma = datasets.toy_dataset_stdev(dataname)
    elif type(dataname) is tuple:
        Y, sigma = dataname
    else:
        raise "dataname has to be a string or tuple"
    kernel =  lambda X,Y : gauss_kernel(X,Y, sigma)
    mmd = lambda X,Y : quadratic_time_mmd(X,Y, kernel)
    statistic = mmd(X,Y)
    p_value = None
    if perms > 0:
        statistics = two_sample_permutation_test(mmd, X, Y, perms, prog_bar=prog_bar)
        p_value = np.mean(statistic <= np.sort(statistics))
    return statistic, p_value

def mix_rbf_kernel(X, Y):
    import shogun as sg
    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.Features(X))
    mmd.set_q(sg.Features(Y))
    mmd.add_kernel(sg.GaussianKernel(10, 1.0))
    mmd.set_kernel_selection_strategy(sg.KSM_MAXIMIZE_MMD)
    mmd.set_train_test_mode(True)       
    mmd.set_train_test_ratio(1)
    mmd.select_kernel()
    statistic = mmd.compute_statistic()
    return statistic

def main():
    #set modules path
    import random
    random.seed(2334)
    samples = 2500
    dataname = "25gaussians"
    X = datasets.toy_dataset(dataname, size=samples)
    print(mmd_toy_metric(X, dataname, samples, perms=10))

if __name__ == "__main__":
    main()