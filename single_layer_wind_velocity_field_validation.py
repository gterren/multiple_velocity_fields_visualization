import pickle, glob, sys, csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from scipy import optimize
from scipy.stats import vonmises, norm, beta, uniform, multivariate_normal
from scipy.special import logsumexp, iv, beta, gamma, digamma
from mpi4py import MPI
from datetime import datetime

from wind_velocity_field_utils import *
from utils import *
from cloud_velocity_vector_utils import *
from feature_extraction_utils import _get_node_info, _save_file


# Remove Duplicate Elements in train and test matrix
def _remove_duplidated(X_tr_, X_ts_, Y_tr_, Y_ts_, W_tr_, W_ts_, N_samples, percentage = 0.75):
    N_tr = Y_tr_.shape[0]
    idx_tr_ = np.arange(N_tr)
    if N_tr > N_samples:
        N_tr = N_samples
        idx_tr_ = np.random.choice(idx_tr_, size = N_samples, replace = False, p = W_tr_/W_tr_.sum())
    _, _, idx_ts_ = np.intersect1d(np.sum(Y_tr_, axis = 1), np.sum(Y_ts_, axis = 1), return_indices = True)
    idx_ts_ = np.delete(np.arange(Y_ts_.shape[0]), idx_ts_)
    N_ts = int(N_tr * percentage)
    if N_ts < idx_ts_.shape[0]:
        idx_ts_ = np.random.choice(idx_ts_, size = N_ts, replace = False, p = W_ts_/W_ts_.sum())
    return X_tr_[idx_tr_, :], X_ts_[idx_ts_, :], Y_tr_[idx_tr_, :], Y_ts_[idx_ts_, :], W_tr_[idx_tr_], W_ts_[idx_ts_]

def _run_experiment(svm, kernel, degree, weights, CV):
    # Start timing
    t_init = datetime.now()
    # Wind velocity estimation by using selected standardized vectors
    theta_ = _wind_velocity_field_svm(XY_tr_, UV_tr_, w_tr_, XY_ts_, UV_ts_, w_ts_, dXYZ_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y,
                                      N_y, N_x, step_size, svm, kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display = False)
    # Stop timing and estimate time for each training and testing
    t = datetime.now() - t_init
    return t.total_seconds(), theta_[2]

def _get_results(time, theta_):
    # Unpack Variables and print results
    cv_, wmse_val, wmse_tr, wmse_ts, mse_tr, mse_ts, wmae_tr, wmae_ts, mae_tr, mae_ts, div, vor = theta_
    print('Time: {} Error: WMSE = {} MSE = {} WMAE = {} MAE = {}'.format(time, wmse_ts, mse_ts, wmae_ts, mae_ts))
    # Variables Initialization
    E_ = np.zeros((10))
    F_ = np.zeros((2))
    P_ = np.zeros((4))
    W_ = np.zeros((4))
    # Save Error-Matrics in a matrix
    E_ = np.array((time, wmse_val, wmse_tr, wmse_ts, mse_tr, mse_ts, wmae_tr, wmae_ts, mae_tr, mae_ts))
    # Save Flow Dynamics in a Matrix
    F_= np.array((div, vor))
    # eSVM CV-Parameters a matrix
    P_ = np.array((cv_[0][0], cv_[0][1], cv_[0][2], cv_[0][3]))
    # In case of two eSVM save then in another matrix
    if len(cv_) == 2:
        W_ = np.array((cv_[1][0], cv_[1][1], cv_[1][2], cv_[1][3]))
    return np.concatenate((E_, F_, P_, W_), axis = 0)

# Save Data in a .csv file
# key - time - WMSE Val - WMSE TR - WMSE TS - MSE TR - MSE TS - WMAE TR - WMAE TS - MAE TR - MAE TS - DIV - VOR - CV00 - CV01 - CV02 - CV03 - CV1 - CV11 - CV12 - CV13
def _save_files(x_, key, name):
    x_ = [key] + x_.tolist()
    print(x_)
    # Save vector of results
    with open(name, 'a', newline = '\n') as f:
       writer = csv.writer(f)
       writer.writerow(x_)

load_path = r'/users/terren/wheeler-scratch/data'
save_path = r'/users/terren/wind_velocity_field/logs'
# Nodes and jobs information for communication from MPI
#i_job, N_jobs, comm = _get_node_info(verbose = False)
# SVM Parameters
i_svm    = int(sys.argv[1])
i_kernel = int(sys.argv[2])
i_data   = int(sys.argv[3])
i_job    = int(sys.argv[4])
weights  = True
CV       = True
step_future_interval = 4
# Set the parameters by cross-validation
kernel_ = ['linear', 'rbf', 'poly', 'poly']
degree_ = [0, 0, 2, 3]
degree = degree_[i_kernel]
kernel = kernel_[i_kernel]
# Cross-Validation Parameters
N_grid  = 5
N_kfold = 3
N_BO_iterations = 15
print(r'Cross-Validation: No Grid Search: {} No. kfolds: {} BO iter.: {}'.format(N_grid, N_kfold, N_BO_iterations))
# Load-up samples
name = r'LK-IS-v4-2.pkl'
file_name = '{}/{}'.format(load_path, name)
X_ = _load_file(file_name)[0]
print(file_name, len(X_))

#for i_svm in [0, 1, 2, 3]:
print(r'Config.: SVM: {} Kernel: {} Degree: {} CV: {}'.format(i_svm, kernel, degree, CV))
#for i_job in range(21):
print(r'Day: {} Sample: {}'.format(i_data, i_job))
# Sample Variables Initializacion
X_1_, _, _, Z_1_, W_1_ = X_[i_data][i_job]
_, X_2_, Y_2_, _, _    = X_[i_data][i_job + step_future_interval]
# Get Training and Test Data
XY_tr_, UV_tr_, w_tr_ = X_1_
XY_ts_, UV_ts_, w_ts_ = X_2_
# Get Prespective Transform Data
XYZ_, dXYZ_, height, wind_flow_indicator_ = Y_2_
# Get Constants
XY_stdz_, xy_stdz_, _stdz_x, _stdz_y    = Z_1_
p_segm, lag, p_sel, N_y, N_x, step_size = W_1_
print('Config.: ', p_segm, lag, p_sel, N_y, N_x, step_size)
# Remove Vectors that are repeated in the training an test set
print(XY_tr_.shape, UV_tr_.shape, w_tr_.shape, XY_ts_.shape, UV_ts_.shape, w_ts_.shape)
XY_tr_, XY_ts_, UV_tr_, UV_ts_, w_tr_, w_ts_ = _remove_duplidated(XY_tr_, XY_ts_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_samples = 250)
print(XY_tr_.shape, UV_tr_.shape, w_tr_.shape, XY_ts_.shape, UV_ts_.shape, w_ts_.shape)
# loop over Samples
time, theta_ = _run_experiment(i_svm, kernel, degree, weights, CV)
x_ = _get_results(time, theta_)
# Save Data only if they have commom wind layers
_save_files(x_, key = i_job, name = r'{}/{}{}-{}-250-100-C1000-nostd.csv'.format(save_path, i_svm, i_kernel, i_data))
