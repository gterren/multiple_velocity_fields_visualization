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

def _run_experiment(svm, kernel, degree, weights, CV):
    # Start timing
    t_init = datetime.now()
    # Wind velocity estimation by using selected standardized vectors
    UV_hat_, theta_ = _wind_velocity_field_svm(XY_tr_, UV_tr_, w_tr_, XY_ts_, UV_ts_, w_ts_, dXYZ_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y,
                                      N_y, N_x, step_size, svm, kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display = False)
    # Stop timing and estimate time for each training and testing
    t = datetime.now() - t_init
    return t.total_seconds(), theta_

def _get_results(time, theta_):
    # Unpack Variables and print results
    cv_, wmse_val, wmse_ts, mse_ts, wmae_ts, mae_ts, div, vor = theta_
    print('Time: {} Error: WMSE = {} MSE = {} WMAE = {} MAE = {}'.format(time, wmse_ts, mse_ts, wmae_ts, mae_ts))
    # Variables Initialization
    E_ = np.zeros((10))
    F_ = np.zeros((2))
    P_ = np.zeros((4))
    W_ = np.zeros((4))
    # Save Error-Matrics in a matrix
    E_ = np.array((time, wmse_val, wmse_ts, mse_ts, wmae_ts, mae_ts))
    # Save Flow Dynamics in a Matrix
    F_= np.array((div, vor))
    # eSVM CV-Parameters a matrix
    P_ = np.array((cv_[0][0], cv_[0][1], cv_[0][2], cv_[0][3]))
    # In case of two eSVM save then in another matrix
    if len(cv_) == 2:
        W_ = np.array((cv_[1][0], cv_[1][1], cv_[1][2], cv_[1][3]))
    return np.concatenate((E_, F_, P_, W_), axis = 0)

# Save Data in a .csv file
# key - time - WMSE Val - WMSE TS- MSE TS - WMAE TS - MAE TS - DIV - VOR - CV00 - CV01 - CV02 - CV03 - CV1 - CV11 - CV12 - CV13
def _save_files(x_, key, name):
    x_ = [key] + x_.tolist()
    print(x_)
    # Save vector of results
    with open(name, 'a', newline = '\n') as f:
       writer = csv.writer(f)
       writer.writerow(x_)

load_path = r'/users/terren/wheeler-scratch/data/DLWVF_testing_v2-28'
save_path = r'/users/terren/wind_velocity_field/logs'
# Nodes and jobs information for communication from MPI
i_job, N_jobs, comm = _get_node_info(verbose = False)
# SVM Parameters
i_svm    = int(sys.argv[1])
i_kernel = int(sys.argv[2])
i_data   = int(sys.argv[3])
i_layer  = int(sys.argv[4])
n_layers = int(sys.argv[5])
weights  = True
CV       = False
step_future_interval = 6
# Set the parameters by cross-validation
kernel_ = ['linear', 'rbf', 'poly', 'poly']
degree_ = [0, 0, 2, 3]
degree = degree_[i_kernel]
kernel = kernel_[i_kernel]
print(r'Config.: SVM: {} Kernel: {} Degree: {} CV: {}'.format(i_svm, kernel, degree, CV))
# Cross-Validation Parameters
N_grid  = 5
N_kfold = 3
N_BO_iterations = 3
print(r'Cross-Validation: No Grid Search: {} No. kfolds: {} BO iter.: {}'.format(N_grid, N_kfold, N_BO_iterations))

# Load-up samples
name = r'D{}L{}-DLWVF-95-6_10100-CV0.pkl'.format(i_data, n_layers)
#name = r'DLWVF_validation_v1/D{}-DLWVF-95-6_10100-CV1.pkl'.format(i_data)
#name = r'D{}-DLWVF-95-8_00110.pkl'.format(i_data)
#name = r'D{}-DLWVF-80-6_02004.pkl'.format(i_data)
file_name = '{}/{}'.format(load_path, name)
X_ = _load_file(file_name)[0]
print(r'Day: {} Wind Layer: {} Sample: {} -- {} File: {} '.format(i_data, i_layer, i_job, len(X_), file_name))
# Sample Variables Initializacion
X_tr_, _, _, Z_tr_, W_tr_ = X_[i_job][i_layer]
_, X_ts_, Y_ts_, _, _     = X_[i_job + step_future_interval][i_layer]
# Get Training and Test Data
XY_tr_, UV_tr_, w_tr_ = X_tr_
XY_ts_, UV_ts_, w_ts_ = X_ts_
# Get Prespective Transform Data
XYZ_, dXYZ_, height, wind_flow_indicator_ = Y_ts_
# Get Constants
XY_stdz_, xy_stdz_, _stdz_x, _stdz_y = Z_tr_
p_segment, n_select, p_train, n_layers, lag, N_y, N_x, step_size = W_tr_
config = r'{}-{}_{}-{}'.format(p_segment, lag, n_select, p_train)
# loop over Samples
time, theta_ = _run_experiment(i_svm, kernel, degree, weights, CV)
x_ = _get_results(time, theta_)
# Save Data only if they have commom wind layers
_save_files(x_, key = r'{}{}'.format(i_job, i_layer),
            name = r'{}/SVM{}{}-95-6_10100-CV{}.csv'.format(save_path, i_svm, i_kernel, int(CV)))
