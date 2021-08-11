import numpy as np

from scipy import optimize
from scipy import sparse

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score, explained_variance_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

from feature_extraction_utils import _divergence, _vorticity
from lib_bayesian_optimization import _bayesian_optimization

from time import time
from datetime import datetime

import math
import torch
import gpytorch

# Defining wind velocity field coordiantes, and also the coordiantes after reducing the dimension
# Plus coordenate system stadardization
def _wind_velocity_field_coordinates(X_, Y_, step_size):
    k = step_size//2
    # Set of coordiantes
    X = X_.flatten()[..., np.newaxis]
    Y = Y_.flatten()[..., np.newaxis]
    # Set of coordiantes after reducing the dimension
    x = X_[k::step_size, k::step_size].flatten()[..., np.newaxis]
    y = Y_[k::step_size, k::step_size].flatten()[..., np.newaxis]
    # Standardize both coordenates system
    x_stdz = preprocessing.StandardScaler().fit(X)
    y_stdz = preprocessing.StandardScaler().fit(Y)
    XY_stdz = np.concatenate((x_stdz.transform(X), y_stdz.transform(Y)), axis = 1)
    xy_stdz = np.concatenate((x_stdz.transform(x), y_stdz.transform(y)), axis = 1)
    return x, y, XY_stdz, xy_stdz, x_stdz, y_stdz

# Reshape prediction outputs
def _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size):
    # Concatenate prediction for multioutput error
    uv_tr_hat_ = np.concatenate((u_tr_hat_[:, np.newaxis], v_tr_hat_[:, np.newaxis]), axis = 1)
    uv_ts_hat_ = np.concatenate((u_ts_hat_[:, np.newaxis], v_ts_hat_[:, np.newaxis]), axis = 1)
    # Reshape from vector to matrix vector velocity components output from eSVM
    U_hat_ = U_hat_.reshape((N_y, N_x))
    V_hat_ = V_hat_.reshape((N_y, N_x))
    u_hat_ = u_hat_.reshape((N_y//step_size, N_x//step_size))
    v_hat_ = v_hat_.reshape((N_y//step_size, N_x//step_size))
    # Concatenate prediction for multioutput error
    UV_hat_ = np.concatenate((U_hat_[..., np.newaxis], V_hat_[..., np.newaxis]), axis = 2)
    uv_hat_ = np.concatenate((u_hat_[..., np.newaxis], v_hat_[..., np.newaxis]), axis = 2)
    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_


def _wind_velocity_field_fit(X_, Y_):
    return preprocessing.StandardScaler().fit(X_[..., np.newaxis]), preprocessing.StandardScaler().fit(Y_[..., np.newaxis])

def _wind_velocity_field_scale(X_, Y_, _stdz_x, _stdz_y):
    return _stdz_x.transform(X_[..., np.newaxis])[:, 0], _stdz_y.transform(Y_[..., np.newaxis])[:, 0]

def _wind_velocity_field_inverse(X_, Y_, _stdz_x, _stdz_y):
    return _stdz_x.inverse_transform(X_), _stdz_y.inverse_transform(Y_)

# Sklearn Epsilon SVM for regression.
def _SK_eSVM_regression(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                        kernel, degree, weights, N_grid, N_kfold, display):

    # Parameters Cross-validation
    def __cross_validation(X_, y_, w_):
        # Set the parameters by cross-validation
        cv_ = [{'kernel': [kernel], 'degree':  [degree], 'C': np.logspace(-2, 2, N_grid), 'epsilon': np.logspace(-3, 1, N_grid),
                'gamma': np.logspace(-2, 2, N_grid), 'coef0': np.logspace(-2, 2, N_grid)}]
        # Configure a grid-search optimization for the parameters validation
        _CV_eSVM = GridSearchCV(SVR(), cv_, scoring = make_scorer(mean_squared_error), cv = N_kfold)
        # Perform the configured optimization
        _CV_eSVM.fit(X_, y_, sample_weight = np.ascontiguousarray(w_))
        return _CV_eSVM.best_params_, _CV_eSVM.best_score_

    # Applying SKlearn method fit from SVR
    def __fit(X_, y_, w_, cv_):
        return SVR(kernel = cv_['kernel'], degree = cv_['degree'], C = cv_['C'], epsilon = cv_['epsilon'],
                   gamma  = cv_['gamma'], coef0 = cv_['coef0'], max_iter = -1).fit(X_, y_, sample_weight = np.ascontiguousarray(w_))

    # Applying SKlearn method predict from eSVR
    def __predict(X_, _eSVM):
        return _eSVM.predict(X_)

    # CV, Train, and predict SVM
    def __train(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, y_tr_, y_ts_, w_tr_, w_ts_, kernel, degree, weights, N_grid, N_kfold, display):
        # w_tr_error_ = w_tr_.copy()
        # if not weights:
        #     w_tr_ = np.ones(w_tr_.shape)/w_tr_.shape[0]
        # Cross-validation of the eSVM and Kernel parametersand
        cv_, WMSE_val = __cross_validation(XY_tr_stdz_, y_tr_, w_tr_)
        if display:
            print('>> CV Paramters: {}'.format(cv_))

        # Fit eSVM with ccoss-validated paramters
        _eSVM = __fit(XY_tr_stdz_, y_tr_, w_tr_, cv_)

        # Return Prediction
        cv_ = [cv_['C'], cv_['epsilon'], cv_['gamma'], cv_['coef0']]

        # Error Analysis
        y_tr_hat_ = __predict(XY_tr_stdz_, _eSVM)
        y_ts_hat_ = __predict(XY_ts_stdz_, _eSVM)

        # Extrapolate Velocity Field to each pixel on a frame
        Y_hat_ = __predict(XY_stdz_, _eSVM)
        y_hat_ = __predict(xy_stdz_, _eSVM)
        return WMSE_val, cv_, Y_hat_, y_hat_, y_tr_hat_, y_ts_hat_

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(UV_tr_[:, 0], UV_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(UV_tr_[:, 0], UV_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(UV_ts_[:, 0], UV_ts_[:, 1], _stdz_u, _stdz_v)
    #U_tr_, V_tr_ = UV_tr_[:, 0], UV_tr_[:, 1]
    #U_ts_, V_ts_ = UV_ts_[:, 0], UV_ts_[:, 1]

    # Train SVM for the U component of the velocity field
    u_WMSE_val, u_cv_, U_hat_, u_hat_, u_tr_hat_, u_ts_hat_ = __train(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, U_tr_, U_ts_, w_tr_, w_ts_,
                                                                     kernel, degree, weights, N_grid, N_kfold, display)

    # Train SVM for the V component of the velocity field
    v_WMSE_val, v_cv_, V_hat_, v_hat_, v_tr_hat_, v_ts_hat_ = __train(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, V_tr_, V_ts_, w_tr_, w_ts_,
                                                                     kernel, degree, weights, N_grid, N_kfold, display)
    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(U_hat_, V_hat_, _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(u_hat_, v_hat_, _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(u_tr_hat_, v_tr_hat_, _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(u_ts_hat_, v_ts_hat_, _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [u_cv_, v_cv_], .5*u_WMSE_val + .5*v_WMSE_val

# Epsilon-SVM single output for regression optimized by SLSQP method
def _eSVM_regression(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                     kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display):

    # Grid Search optimization method
    def __cross_validation(X_tr_, y_tr_, w_tr_, theta_):
        # Training a model for a cross-validation iteration
        def ___CV(X_tr_, X_val_, y_tr_, y_val_, w_tr_, w_val_, C, epsilon, beta, gamma):
            # Validataion kernel for training and test
            K_tr_  = __kernel(X_tr_,  X_tr_, kernel, degree, gamma, beta)
            K_val_ = __kernel(X_tr_, X_val_, kernel, degree, gamma, beta)
            # Fiting eSVM model
            a_, a_star_, b, rSVs_ = __fit(K_tr_, y_tr_, w_tr_, C, epsilon)
            # Calculating the prediction error
            y_val_hat_ = np.nan_to_num(__predict(K_val_, a_star_, b))
            return mean_squared_error(y_val_, y_val_hat_, sample_weight = w_val_, multioutput = 'uniform_average')
        # Implementation of K-fold cross-validation
        def ___KF(arg_, X_tr_, y_tr_, w_tr_):
            # Obtaining only values for the variables optimzied
            def ____interface(arg_, x_0 = 0, x_1 = 0, x_2 = 0, x_3 = 0):
                x_0 = 10**arg_[0]
                x_1 = 10**arg_[1]
                if len(arg_) > 2: x_2 = 10**arg_[2]
                if len(arg_) > 3: x_3 = 10**arg_[3]
                return x_0, x_1, x_2, x_3
            C, epsilon, gamma, beta = ____interface(np.squeeze(arg_))
            e_ = np.zeros((N_kfold))
            k  = 0
            #print(C, epsilon, gamma, beta)
            for idx_tr_, idx_val_ in KFold(n_splits = N_kfold, random_state = None, shuffle = False).split(X_tr_):
               e_[k] = ___CV(X_tr_[idx_tr_, :], X_tr_[idx_val_, :], y_tr_[idx_tr_], y_tr_[idx_val_], w_tr_[idx_tr_], w_tr_[idx_val_], C, epsilon, beta, gamma)
               k += 1
            if np.isnan(e_.mean()):
                return 1e10
            else:
                return e_.mean()
        # Adapt parameters dictionary to GS optimization function
        def ___int_bounds(theta_, N_grid, bounds_ = ((1e-2, 1e2), (1e-3, 1e1), (1e-3, 1e1), (1e-2, 1e2))):
            log_bounds_ = []
            for i in range(len(theta_)):
                if theta_[i] != 0.0:
                    log_bounds_.append((np.log10(bounds_[i][0]), np.log10(bounds_[i][1])))
            return log_bounds_
        # Optima to logarithmic form
        def ___int_optima(opt_, x_ = [0., 0., 0., 0.]):
            for i in range(len(opt_)): x_[i] = 10**opt_[i]
            return x_
        # Adapt parameters dictionary to BO optimization function
        def ___int_bounds_bo(theta_, list_ = ()):
            for _, i in zip( range(len(theta_)), ['C', 'epsilon', 'gamma', 'coef0']):
                list_.append(range(theta_[i][0], theta_[i][1]))
            return np.array(list_)
        # Initial poings in the BO are those evaluated in the Grid Search
        def ___int_init(opt_):
            X_, e_, XY_, Z_ = opt_
            XYZ_ = Z_.flatten()[:, np.newaxis]
            for i in range(XY_.shape[0]):
                XYZ_ = np.concatenate((XYZ_, XY_[i, ...].flatten()[:, np.newaxis]), axis = 1)
            return XYZ_

        # Optimization bounds defintion
        bounds_ = ___int_bounds(theta_, N_grid)
        # Grid Search Optimization
        opt_ = optimize.brute(___KF, bounds_, args = (X_tr_, y_tr_, w_tr_), Ns = N_grid, full_output = True, finish = None)
        # Bayesian Optimization
        opt_ = _bayesian_optimization(___KF, bounds_, _aqf = 'EI', xi = 0.01, kappa = 5.,
                                      n_initial_points = ___int_init(opt_), args_ = (X_tr_, y_tr_, w_tr_), n_iterations = N_BO_iterations, display = display)
        return ___int_optima(opt_[0]), opt_[1]

    # Optimal parameters per Kernel
    def __optimal_parameters():
        # 36.68 0.33 0 0
        # 87.39 0.19 5.20 0
        # 33.11 0.31 4.78 24.45
        # 31.69 0.20 2.64 17.55
        if kernel is 'linear' and degree is 0: return [36.68, 0.33, 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [87.39, 0.19, 5.20, 0.]
        if kernel is 'poly'   and degree is 2: return [33.11, 0.31, 4.78, 24.45]
        if kernel is 'poly'   and degree is 3: return [31.69, 0.20, 2.64, 17.55]

    # Epsilon-SVM model fit...
    def __fit(K_, y_, w_, C, epsilon, tau = 1e-6):
        # Solving optimization problem to calculate the dual parameters
        def ___dual(K_, y_, w_, C, epsilon):
            # Defining contrains and their derivatives
            def ___f(a_):   return .5 * a_.T @ H_ @ a_ + a_.T @ f_
            def ___df(a_):  return H_ @ a_ + f_
            def ___ddf(a_): return H_
            def ___c_eq(a_):  return A_eq_ @ a_ - b_eq_ @ o_
            def ___dc_eq(a_): return A_eq_
            # Variable Initialization
            N  = K_.shape[0]
            o_ = np.ones((2*N,))
            w_ = np.concatenate((w_, w_), axis = 0)
            a_0_ = np.random.uniform(0, C, 2*N)
            # Defining Least-Square Problem
            f_ = np.concatenate((epsilon - y_, epsilon + y_), axis = 0)
            H_ = np.concatenate((np.concatenate((K_, -K_), axis = 0), np.concatenate((-K_, K_), axis = 0)), axis = 1)
            # Equality constrains
            A_eq_ = np.squeeze(np.concatenate( (np.ones((1, N)), - np.ones((1, N)) ), axis = 1))
            b_eq_ = (w_ * C * o_)/(2 * N)
            # Defining constrains and boundaries for contrained-optimization problem
            constr_ = {'type':'eq', 'fun': ___c_eq,'jac': ___dc_eq}
            bounds_ = np.concatenate( ( np.zeros((2*N, 1)), C * w_[:, np.newaxis] * np.ones((2*N, 1)) ), axis = 1)
            # solving for dual paramaters
            OPT_ = optimize.minimize(___f, a_0_, jac = ___df, bounds = bounds_,  constraints = constr_, method = 'SLSQP',
                                     options = {'maxiter': 25, 'ftol': C/1000., 'disp': False})
            # Calculaing and returning alpha parameters
            return OPT_['x'], OPT_['x'][:N] - OPT_['x'][N:]
        # Solving linear system to find out the bias
        def ___bias(K_, y_, w_, alpha_, C, epsilon, tau):
            # Find the Support Vectors
            SVs_ = np.where(np.squeeze((alpha_ > tau) & (alpha_ < (w_ * C) - tau)) == True)[0]
            # Cacluclate Bias when there are SVs and when there are not!
            if len(SVs_) > 0: bias = y_[SVs_] - K_[SVs_, :] @ alpha_ - epsilon
            else:             bias = y_       - K_ @ alpha_          - epsilon
            # Return the average bias for each Support Vector
            return bias.mean(), SVs_
        # Solve quadratic problem to obtain the dual paramters
        a_, a_star_ = ___dual(K_, y_, w_, C, epsilon)
        # Solve Least-Square problem to obtain the bias using the Support Vectros
        b, iSVs_ = ___bias(K_, y_, w_, a_star_, C, epsilon, tau)
        return a_, a_star_, b, iSVs_

    # Calculing prediction for a new sample
    def __predict(K_, alpha_, bias): return K_.T @ alpha_ + bias

    # Implementation of Kernel Functions...
    def __kernel(X, Y, kernel, degree, gamma = 0., beta = 0.):
        if kernel is 'linear': return X @ Y.T
        if kernel is 'poly':   return (gamma * X @ Y.T + beta)**degree
        if kernel is 'rbf':    return np.exp(- gamma * euclidean_distances(X, Y)**2)

    # CV, Train, and predict SVM
    def __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, y_tr_, y_ts_, w_tr_, w_ts_,
                kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display, UV):
        # CV-parameters for the x-component
        cv_, WMSE_val = __optimal_parameters(), 0.
        if CV: cv_, WMSE_val = __cross_validation(XY_stdz_tr_, y_tr_, w_tr_, cv_)

        # Evaluate kernel function for training and test
        K_tr_  = __kernel(XY_stdz_tr_, XY_stdz_tr_, kernel, degree, gamma = cv_[2], beta = cv_[3])
        K_ts_  = __kernel(XY_stdz_tr_, XY_stdz_ts_, kernel, degree, gamma = cv_[2], beta = cv_[3])
        K_hat_ = __kernel(XY_stdz_tr_,    XY_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])
        k_hat_ = __kernel(XY_stdz_tr_,    xy_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])
        # Fit model for the x-velocity component
        a_, a_star_, b, iSVs_ = __fit(K_tr_, y_tr_, w_tr_, C = cv_[0], epsilon = cv_[1], tau = 1e-6)

        # Error Analysis
        y_tr_hat_ = __predict(K_tr_, a_star_, b)
        y_ts_hat_ = __predict(K_ts_, a_star_, b)

        # Extrapolate Velocity Field to each pixel on a frame
        Y_hat_ = __predict(K_hat_, a_star_, b)
        y_hat_ = __predict(k_hat_, a_star_, b)

        return WMSE_val, cv_, Y_hat_, y_hat_, y_tr_hat_, y_ts_hat_

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(UV_tr_[:, 0], UV_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(UV_tr_[:, 0], UV_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(UV_ts_[:, 0], UV_ts_[:, 1], _stdz_u, _stdz_v)
    # U_tr_, V_tr_ = UV_tr_[:, 0], UV_tr_[:, 1]
    # U_ts_, V_ts_ = UV_ts_[:, 0], UV_ts_[:, 1]

    # Train SVM for the U component of the velocity field
    u_WMSE_val, u_cv_, U_hat_std_, u_hat_std_, u_tr_hat_std_, u_ts_hat_std_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, U_tr_, U_ts_, w_tr_, w_ts_,
                                                                                      kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display, UV = 'U')
    # Train SVM for the V component of the velocity field
    v_WMSE_val, v_cv_, V_hat_std_, v_hat_std_, v_tr_hat_std_, v_ts_hat_std_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, V_tr_, V_ts_, w_tr_, w_ts_,
                                                                                      kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display, UV = 'V')

    # u_WMSE_val, u_cv_, U_hat_, u_hat_, u_tr_hat_, u_ts_hat_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, U_tr_, U_ts_, w_tr_, w_ts_,
    #                                                                                   kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display, UV = 'U')
    # # Train SVM for the V component of the velocity field
    # v_WMSE_val, v_cv_, V_hat_, v_hat_, v_tr_hat_, v_ts_hat_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, V_tr_, V_ts_, w_tr_, w_ts_,
    #                                                                                   kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display, UV = 'V')

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(U_hat_std_, V_hat_std_, _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(v_hat_std_, v_hat_std_, _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(u_tr_hat_std_, v_tr_hat_std_, _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(u_ts_hat_std_, v_ts_hat_std_, _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [u_cv_, v_cv_], .5*u_WMSE_val + .5*v_WMSE_val

# Epsilon-SVM multiple output for regression optimized by SLSQP method
def _MO_eSVM_regression(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, Y_tr_, Y_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                        kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display  = None):
    # Grid Search optimization method
    def __cross_validation(X_tr_, Y_tr_, w_tr_, theta_):
        # Training a model for a cross-validation iteration
        def ___CV(X_tr_, X_val_, Y_tr_, Y_val_, w_tr_, w_val_, C, epsilon, beta, gamma):
            # Validataion kernel for training and test
            K_tr_  = __kernel(X_tr_,  X_tr_, kernel, degree, gamma, beta)
            K_val_ = __kernel(X_tr_, X_val_, kernel, degree, gamma, beta)
            # Fiting eSVM model
            a_, a_star_, b_, iSVs_ = __fit(K_tr_, Y_tr_, w_tr_, C, epsilon)
            Y_val_hat_ = np.nan_to_num(__predict(K_val_, a_star_, b_))
            # Calculating the prediction error
            u_WMSE_val = mean_squared_error(Y_val_[:, 0], Y_val_hat_[:, 0], sample_weight = w_val_, multioutput = 'uniform_average')
            v_WMSE_val = mean_squared_error(Y_val_[:, 1], Y_val_hat_[:, 1], sample_weight = w_val_, multioutput = 'uniform_average')
            return .5*u_WMSE_val + .5*v_WMSE_val
        # Implementation of K-fold cross-validation
        def ___KF(arg_, X_tr_, Y_tr_, w_tr_):
            # Obtaining only values for the variables optimzied
            def ____interface(arg_, x_0 = 0, x_1 = 0, x_2 = 0, x_3 = 0):
                x_0 = 10**arg_[0]
                x_1 = 10**arg_[1]
                if len(arg_) > 2: x_2 = 10**arg_[2]
                if len(arg_) > 3: x_3 = 10**arg_[3]
                return x_0, x_1, x_2, x_3
            C, epsilon, gamma, beta = ____interface(np.squeeze(arg_))
            e_ = np.zeros((N_kfold))
            k  = 0
            for idx_tr_, idx_val_ in KFold(n_splits = N_kfold, random_state = None, shuffle = False).split(X_tr_):
                e_[k] = ___CV(X_tr_[idx_tr_, :], X_tr_[idx_val_, :], Y_tr_[idx_tr_], Y_tr_[idx_val_], w_tr_[idx_tr_], w_tr_[idx_val_], C, epsilon, beta, gamma)
                k += 1
            if np.isnan(e_.mean()):
                return 1e10
            else:
                return e_.mean()
        # Adapt parameters dictionary to GS optimization function
        def ___int_bounds(theta_, N_grid, bounds_ = ((1e-2, 1e2), (1e-3, 1e1), (1e-3, 1e1), (1e-2, 1e2))):
            log_bounds_ = []
            for i in range(len(theta_)):
                if theta_[i] != 0.0:
                    log_bounds_.append((np.log10(bounds_[i][0]), np.log10(bounds_[i][1])))
            return log_bounds_
        # Optima to logarithmic form
        def ___int_optima(opt_, x_ = [0., 0., 0., 0.]):
            for i in range(len(opt_)): x_[i] = 10**opt_[i]
            return x_
        # Adapt parameters dictionary to BO optimization function
        def ___int_bounds_bo(theta_, list_ = ()):
            for _, i in zip( range(len(theta_)), ['C', 'epsilon', 'gamma', 'coef0']):
                list_.append(range(theta_[i][0], theta_[i][1]))
            return np.array(list_)
        # Initial poings in the BO are those evaluated in the Grid Search
        def ___int_init(opt_):
            X_, e_, XY_, Z_ = opt_
            XYZ_ = Z_.flatten()[:, np.newaxis]
            for i in range(XY_.shape[0]):
                XYZ_ = np.concatenate((XYZ_, XY_[i, ...].flatten()[:, np.newaxis]), axis = 1)
            return XYZ_

        # Optimization bounds defintion
        bounds_ = ___int_bounds(theta_, N_grid)
        # Grid-Search Optimization
        opt_ = optimize.brute(___KF, bounds_, args = (X_tr_, Y_tr_, w_tr_), Ns = N_grid, full_output = True, finish = None)
        # Bayesian Optimization
        opt_ = _bayesian_optimization(___KF, bounds_, _aqf = 'EI', xi = 0.01, kappa = 5.,
                                      n_initial_points = ___int_init(opt_), args_ = (X_tr_, Y_tr_, w_tr_), n_iterations = N_BO_iterations, display = display)
        return ___int_optima(opt_[0]), opt_[1]

    # Saved optimal parameters
    def __optimal_parameters():
        # 31.06 0.31 0 0
        # 38.71 0.36 17.81 0
        # 34.73 0.28 4.65 10.66
        # 38.47 0.22 4.19 2.33
        if kernel is 'linear' and degree is 0: return [31.06, 0.31, 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [8.71, 0.36, 17.81, 0.]
        if kernel is 'poly'   and degree is 2: return [34.73, 0.28, 4.65, 10.66]
        if kernel is 'poly'   and degree is 3: return [38.47, 0.22, 4.19, 2.33]

    # Epsilon-SVM model fit...
    def __fit(K_tilde_, Y_, w_, C, epsilon, tau = 1e-5):
        # Solving optimization problem to calculate the dual parameters
        def ___dual(K_tilde_, Y_, w_, C, epsilon):
            # Defining contrains and its derivations
            def ___f(a_):   return .5 * a_.T @ H_tilde_ @ a_ + a_.T @f_tilde_
            def ___df(a_):  return H_tilde_ @ a_ + f_tilde_.T
            def ___ddf(a_): return H_tilde_
            def ___c_eq(a_):  return A_eq_ @ a_ - b_eq_ @ o_
            def ___dc_eq(a_): return A_eq_
            # Variable Initialization
            N, T = Y_.shape
            o_ = np.ones((2*T*N,))
            w_tilde_ = np.concatenate((w_, w_, w_, w_), axis = 0)
            a_0_ = np.random.uniform(0, C, 2*T*N)
            # Defining Least-Square Problem
            y_tilde_ = np.reshape(Y_, N*T, order = 'F')
            f_tilde_ = np.concatenate((epsilon - y_tilde_, epsilon + y_tilde_), axis = 0)
            H_tilde_ = sparse.csr_matrix(np.concatenate((np.concatenate((K_tilde_, -K_tilde_), axis = 0), np.concatenate((-K_tilde_, K_tilde_), axis = 0)), axis = 1))
            # Equality constrains
            A_eq_ = np.concatenate( (np.ones((1, T*N)), - np.ones((1, T*N))), axis = 1)
            b_eq_ = (w_tilde_ * C * o_)/(2 * T * N)
            # Defining constrain-opotimization problem and solving for dual paramaters
            constr_ = {'type':'eq', 'fun': ___c_eq, 'jac': ___dc_eq}
            bounds_ = np.concatenate((np.zeros((2*T*N, 1)), (w_tilde_[:, np.newaxis] * C) * np.ones((2*T*N, 1))), axis = 1)
            OPT_ = optimize.minimize(___f, a_0_, jac = ___df, bounds = bounds_,  constraints = constr_, method = 'SLSQP',
                                     options = {'maxiter': 25, 'ftol': C/1000., 'disp': False})
            # Calculating and returning alpha parameters
            return OPT_['x'], OPT_['x'][:T*N] - OPT_['x'][T*N:]
        # Solving linear system to find out the bias
        def ___bias(K_tilde_, Y_, w_, alpha_, C, epsilon, tau):
            # Constant definition
            N, T = Y_.shape
            y_tilde_ = np.reshape(Y_, N*T, order = 'F')
            # Variables Initialization
            bias_, SVs_, idx_ = np.zeros((2, 1)), [], np.arange(N*T)
            # Loop over the multiple ouputs to estimate independent bias
            for i in range(T):
                # Find Support Vectors
                SV_ = np.where(np.squeeze((alpha_[i*N:(i + 1)*N] > tau) & (alpha_[i*N:(i + 1)*N] < (w_ * C) - tau)) == True)[0]
                # Solve Least-Square problem to obtain the bias using the Support Vectros
                if len(SV_) > 0: b = y_tilde_[idx_[i*N:(i + 1)*N]][SV_] - K_tilde_[idx_[i*N:(i + 1)*N][SV_], :] @ alpha_ - epsilon
                else:            b = y_tilde_[idx_[i*N:(i + 1)*N]]      - K_tilde_[idx_[i*N:(i + 1)*N], :]      @ alpha_ - epsilon
                # Average bias and Support Vectors per output variable
                bias_[i,:] = b.mean()
                SVs_.append(SV_)
            # Return the average bias for each Support Vector
            return bias_, SVs_
        # Solve quadratic problem to obtain the dual paramters
        a_, a_star_ = ___dual(K_tilde_, Y_, w_, C, epsilon)
        # Solve Least-Square problem to obtain the bias using the Support Vectros
        b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
        return a_, a_star_, b_, iSVs_

    # Calculing prediction for new sample
    def __predict(K_tilde_, a_star_, b_):
        y_tilde_hat_ = K_tilde_.T @ a_star_
        N = y_tilde_hat_.shape[0]//2
        return np.concatenate((y_tilde_hat_[:N][:, np.newaxis], y_tilde_hat_[N:][:, np.newaxis]), axis = 1) + b_.T

    # Multi-output Kernels
    def __kernel(X_, Y_, kernel, degree, beta = None, gamma = None, independent = 0):
        # Implementation of Kernel Functions...
        def ___K(X, Y, beta, gamma, kernel, degree):
            if kernel is 'linear': return X @ Y.T
            if kernel is 'linear_dependent': return gamma*X @ Y.T
            if kernel is 'poly':   return (gamma*X @ Y.T + beta)**degree
            if kernel is 'rbf':    return np.exp(-gamma*euclidean_distances(X, Y)**2)
        # Autocorrelation Zero
        O_ = np.zeros((X_.shape[0], Y_.shape[0]))
        if independent == 1:
            x_ = X_[..., 0][:, np.newaxis]
            y_ = X_[..., 1][:, np.newaxis]
            x_star_ = Y_[..., 0][:, np.newaxis]
            y_star_ = Y_[..., 1][:, np.newaxis]
            K_11_ = ___K(x_, x_star_, beta, gamma, kernel, degree)
            #K_12_ = ___K(x_, y_star_, beta, gamma, kernel, degree)
            #K_21_ = ___K(y_, x_star_, beta, gamma, kernel, degree)
            K_22_ = ___K(y_, y_star_, beta, gamma, kernel, degree)
            K_tilde_ = np.concatenate((np.concatenate((K_11_, O_), axis = 0), np.concatenate((O_, K_22_), axis = 0)), axis = 1)
            #K_tilde_ = np.concatenate((np.concatenate((K_11_, -K_12_), axis = 0), np.concatenate((K_21_, K_22_), axis = 0)), axis = 1)
        else:
            # Kernel submatrix for Grand Matrix
            K_1_ = ___K(X_, Y_, beta, gamma, kernel, degree)
            K_2_ = ___K(X_, Y_, beta, gamma, kernel, degree)
            K_tilde_ = np.concatenate((np.concatenate((K_1_, O_), axis = 0), np.concatenate((O_, K_2_), axis = 0)), axis = 1)
        # Grand Matrix construct out of kernel submatrix and 0 matrix
        return K_tilde_

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(Y_tr_[:, 0], Y_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(Y_tr_[:, 0], Y_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(Y_ts_[:, 0], Y_ts_[:, 1], _stdz_u, _stdz_v)
    Y_tr_ = np.concatenate((U_tr_[:, np.newaxis], V_tr_[:, np.newaxis],), axis  = 1)
    Y_ts_ = np.concatenate((U_ts_[:, np.newaxis], V_ts_[:, np.newaxis],), axis  = 1)

    # CV-parameters for x, y components
    cv_, WMSE_val = __optimal_parameters(), 0.
    if CV: cv_, WMSE_val = __cross_validation(XY_stdz_tr_, Y_tr_, w_tr_, cv_)

    # Evaluate kernel function for training and testing
    K_tr_  = __kernel(XY_stdz_tr_, XY_stdz_tr_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    K_ts_  = __kernel(XY_stdz_tr_, XY_stdz_ts_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    K_hat_ = __kernel(XY_stdz_tr_,    XY_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    k_hat_ = __kernel(XY_stdz_tr_,    xy_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])

    # Fit model for the x,y-velocity components
    a_, a_star_, b_, iSVs_ = __fit(K_tr_, Y_tr_, w_tr_, C = cv_[0], epsilon = cv_[1], tau = 1e-6)

    # Error Analysis
    uv_tr_hat_ = __predict(K_tr_, a_star_, b_)
    uv_ts_hat_ = __predict(K_ts_, a_star_, b_)
    # Extrapolate Velocity Field to each pixel on a frame
    UV_hat_ = __predict(K_hat_, a_star_, b_)
    uv_hat_ = __predict(k_hat_, a_star_, b_)

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(UV_hat_[:, 0], UV_hat_[:, 1], _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(uv_hat_[:, 0], uv_hat_[:, 1], _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(uv_tr_hat_[:, 0], uv_tr_hat_[:, 1], _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(uv_ts_hat_[:, 0], uv_ts_hat_[:, 1], _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    # Reshape Prediction
    #UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(UV_hat_[..., 0], UV_hat_[..., 1], uv_hat_[..., 0], uv_hat_[..., 1], uv_tr_hat_[..., 0], uv_tr_hat_[..., 1], uv_ts_hat_[..., 0], uv_ts_hat_[..., 1], N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [cv_], WMSE_val

# Epsilon-SVM multiple output with fluid machanics contrains for regression optimized by SLSQP method
def _MO_eSVM_flow_constrains(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, Y_tr_, Y_ts_, w_tr_, w_ts_, dXYZ_, N_y, N_x, step_size,
                             kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display):
    # Grid Search optimization method
    def __cross_validation(X_tr_, X_stdz_, Y_tr_, w_tr_, dXYZ_, theta_):
        # Training a model for a cross-validation iteration
        def ___CV(X_tr_, X_val_,  X_stdz_, Y_tr_, Y_val_, w_tr_, w_val_, dXYZ_, C, epsilon, beta, gamma):
            # Validataion kernel for training and test
            K_tr_  = __kernel(X_tr_,   X_tr_, kernel, degree, gamma, beta)
            K_val_ = __kernel(X_tr_,  X_val_, kernel, degree, gamma, beta)
            K_hat_ = __kernel(X_tr_, X_stdz_, kernel, degree, gamma, beta)
            m_, s_, S_ = _standardization(N_y, N_x, n = K_hat_.shape[0], m_x = _stdz_u.mean_, s_x = _stdz_u.var_, m_y = _stdz_v.mean_, s_y = _stdz_v.var_)
            # Fiting eSVM model
            a_, a_star_, b_, iSVs_ = __fit(K_tr_, K_hat_, Y_tr_, w_tr_, dXYZ_, [m_, s_, S_], C, epsilon)
            Y_val_hat_ = np.nan_to_num(__predict(K_val_, a_star_, b_))
            # Calculating the prediction error
            u_WMSE_val = mean_squared_error(Y_val_[:, 0], Y_val_hat_[:, 0], sample_weight = w_val_ + 1e-5, multioutput = 'uniform_average')
            v_WMSE_val = mean_squared_error(Y_val_[:, 1], Y_val_hat_[:, 1], sample_weight = w_val_ + 1e-5, multioutput = 'uniform_average')
            return .5*u_WMSE_val + .5*v_WMSE_val
        # Implementation of K-fold cross-validation
        def ___KF(arg_, X_tr_, X_stdz_, Y_tr_, w_tr_, dXYZ_):
            # Obtaining only values for the variables optimzied
            def ____interface(arg_, x_0 = 0, x_1 = 0, x_2 = 0, x_3 = 0):
                x_0 = 10**arg_[0]
                x_1 = 10**arg_[1]
                if len(arg_) > 2: x_2 = 10**arg_[2]
                if len(arg_) > 3: x_3 = 10**arg_[3]
                return x_0, x_1, x_2, x_3
            C, epsilon, gamma, beta = ____interface(np.squeeze(arg_))
            e_ = np.zeros((N_kfold))
            k  = 0
            for idx_tr_, idx_val_ in KFold(n_splits = N_kfold, random_state = None, shuffle = False).split(X_tr_):
                e_[k] = ___CV(X_tr_[idx_tr_, :], X_tr_[idx_val_, :],  X_stdz_, Y_tr_[idx_tr_], Y_tr_[idx_val_], w_tr_[idx_tr_], w_tr_[idx_val_], dXYZ_, C, epsilon, beta, gamma)
                k += 1
            if np.isnan(e_.mean()):
                return 1e10
            else:
                return e_.mean()
        # Adapt parameters dictionary to GS optimization function
        def ___int_bounds(theta_, N_grid, bounds_ = ((1e-2, 1e2), (1e-3, 1e1), (1e-3, 1e1), (1e-2, 1e2))):
            log_bounds_ = []
            for i in range(len(theta_)):
                if theta_[i] != 0.0:
                    log_bounds_.append((np.log10(bounds_[i][0]), np.log10(bounds_[i][1])))
            return log_bounds_
        # Optima to logarithmic form
        def ___int_optima(opt_, x_ = [0., 0., 0., 0.]):
            for i in range(len(opt_)): x_[i] = 10**opt_[i]
            return x_
        # Adapt parameters dictionary to BO optimization function
        def ___int_bounds_bo(theta_, list_ = ()):
            for _, i in zip( range(len(theta_)), ['C', 'epsilon', 'gamma', 'coef0']):
                list_.append(range(theta_[i][0], theta_[i][1]))
            return np.array(list_)
        # Initial poings in the BO are those evaluated in the Grid Search
        def ___int_init(opt_):
            X_, e_, XY_, Z_ = opt_
            XYZ_ = Z_.flatten()[:, np.newaxis]
            for i in range(XY_.shape[0]):
                XYZ_ = np.concatenate((XYZ_, XY_[i, ...].flatten()[:, np.newaxis]), axis = 1)
            return XYZ_

        # Optimization Bounds Defition
        bounds_ = ___int_bounds(theta_, N_grid)
        # Grid-Search Optimization
        opt_ = optimize.brute(___KF, bounds_, args = (X_tr_, X_stdz_, Y_tr_, w_tr_, dXYZ_), Ns = N_grid, full_output = True, finish = None)
        # Bayesian Optimization
        opt_ = _bayesian_optimization(___KF, bounds_, _aqf = 'EI', xi = 0.01, kappa = 5.,
                                      n_initial_points = ___int_init(opt_), args_ = (X_tr_, X_stdz_, Y_tr_, w_tr_, dXYZ_),
                                      n_iterations = N_BO_iterations, display = display)
        return ___int_optima(opt_[0]), opt_[1]

    # Saved optimal parameters
    def __optimal_parameters():
        # 38.50 0.19 0 0
        # 38.52 0.35 13.92 0
        # 39.72 0.24 3.78 44.8
        # 12.88 0.22 5.61 8.34
        if kernel is 'linear' and degree is 0: return [38.50, 0.19, 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [38.52, 0.35, 13.92, 0.]
        if kernel is 'poly'   and degree is 2: return [39.72, 0.24, 3.78, 44.8]
        if kernel is 'poly'   and degree is 3: return [12.88, 0.22, 5.61, 8.34]

    # Epsilon-SVM model fit...
    def __fit(K_tilde_, K_hat_, Y_, w_, dXYZ_, std_, C, epsilon, tau = 1e-5, tol = 1e-3):
        # Solving optimization problem to calculate the dual parameters
        def ___dual(K_tilde_, K_hat_, Y_, w_, dXYZ_, std_, C, epsilon):
            # Defining contrains and its derivations
            def ___f(a_):   return 0.5 * a_.T @ H_tilde_ @ a_ + a_.T @ f_tilde_
            def ___df(a_):  return H_tilde_ @ a_ + f_tilde_.T
            def ___ddf(a_): return H_tilde_
            def ___c0_eq(a_):  return A_eq_ @ a_ - b_eq_ @ o_
            def ___dc0_eq(a_): return A_eq_[0, :]
            # Fluid Mechanics Constrains...
            # Calculating Divergence Constrain
            def ___c1_eq(a_):
                a_star_ = a_dot_.T @ a_
                b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
                y_hat_ = K_hat_.T @ a_star_
                bias_  = np.concatenate((np.ones(y_hat_.shape[0]//2)*b_[0, 0], np.ones(y_hat_.shape[0]//2)*b_[1, 0]), axis = 0)*s_ + m_
                Y_hat_ = y_hat_ + bias_
                div = Y_hat_.T @ dD_ @ Y_hat_
                return div
            def ___dc1_eq(a_):
                a_star_ = a_dot_.T @ a_
                b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
                G_ = a_dot_ @ K_hat_
                y_hat_ = G_.T @ a_
                bias_  = np.concatenate((np.ones(y_hat_.shape[0]//2)*b_[0, 0], np.ones(y_hat_.shape[0]//2)*b_[1, 0]), axis = 0)*s_ + m_
                return 2.* G_ @ dD_ @ y_hat_ - 2.* G_ @ dD_ @ bias_
            # Calculating Vorticy Constrain
            def ___c2_eq(a_):
                a_star_ = a_dot_.T @ a_
                b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
                y_hat_ = K_hat_.T @ a_star_
                bias_  = np.concatenate((np.ones(y_hat_.shape[0]//2)*b_[0, 0], np.ones(y_hat_.shape[0]//2)*b_[1, 0]), axis = 0)*s_ + m_
                Y_hat_ = y_hat_ + bias_
                div = Y_hat_.T @ dV_ @ Y_hat_
                #print('Div', div)
                return div
            def ___dc2_eq(a_):
                #return a_.T @ ndKdDiv_
                a_star_ = a_dot_.T @ a_
                b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
                G_ = a_dot_ @ K_hat_
                y_hat_ = G_.T @ a_
                bias_  = np.concatenate((np.ones(y_hat_.shape[0]//2)*b_[0, 0], np.ones(y_hat_.shape[0]//2)*b_[1, 0]), axis = 0)*s_ + m_
                return 2.* G_ @ dV_ @ y_hat_ - 2.* G_ @ dV_ @ bias_
            m_, s_, S_ = std_
            # Variable Initialization
            N, T = Y_.shape
            o_ = np.ones((2*T*N,))
            w_tilde_ = np.concatenate((w_, w_, w_, w_), axis = 0)
            # SVM Variables Definition
            a_0_ = np.random.uniform(0, C, 2*T*N)#[:, np.newaxis]
            bounds_ = np.concatenate((np.zeros((2*T*N, 1)), (w_tilde_[:, np.newaxis] * C) * np.ones((2*T*N, 1))), axis = 1)
            # Alpha substraction Operator
            a_dot_ = np.concatenate((np.eye(N*T), - np.eye(N*T)), axis = 0)
            # Defining Least-Square Problem
            y_tilde_ = np.reshape(Y_, N*T, order = 'F')
            f_tilde_ = np.concatenate((epsilon - y_tilde_, epsilon + y_tilde_), axis = 0)
            H_tilde_ = sparse.csr_matrix(np.concatenate((np.concatenate((K_tilde_, -K_tilde_), axis = 0), np.concatenate((-K_tilde_, K_tilde_), axis = 0)), axis = 1))
            # Equality constrains
            A_eq_ = np.concatenate((np.ones((1, T*N)), - np.ones((1, T*N))), axis = 1)
            b_eq_ = (w_tilde_* C * o_)/(2 * T * N)
            # Defining constrain-opotimization problem and solving for dual paramaters
            K_hat_ = K_hat_*S_
            # Defining constrain-opotimization problem and solving for dual paramaters
            constr_ = [{'type':'eq', 'fun': ___c0_eq, 'jac': ___dc0_eq},
                       {'type':'eq', 'fun': ___c1_eq, 'jac': ___dc1_eq},
                       {'type':'eq', 'fun': ___c2_eq, 'jac': ___dc2_eq}]
            OPT_ = optimize.minimize(fun = ___f, jac = ___df, x0 = a_0_, bounds = bounds_,  constraints = [{'type':'eq', 'fun': ___c0_eq, 'jac': ___dc0_eq}], method = 'SLSQP',
                                     options = {'maxiter': 25, 'ftol': C/1000., 'disp': False})
            OPT_ = optimize.minimize(fun = ___f, jac = ___df, x0 = OPT_['x'], bounds = bounds_,  constraints = constr_, method = 'SLSQP',
                                     options = {'maxiter': 25, 'ftol': C/1000., 'disp': False})
            # Calculating and returning alpha parameters
            return OPT_['x'], OPT_['x'][:T*N] - OPT_['x'][T*N:]
        # Solving linear system to find out the bias
        def ___bias(K_tilde_, Y_, w_, alpha_, C, epsilon, tau):
            # Constant definition
            N, T = Y_.shape
            y_tilde_ = np.reshape(Y_, N*T, order = 'F')
            # Variables Initialization
            bias_, SVs_, idx_ = np.zeros((2, 1)), [], np.arange(N*T)
            # Loop over the multiple ouputs to estimate independent bias
            for i in range(T):
                # Find Support Vectors
                SV_ = np.where(np.squeeze((alpha_[i*N:(i + 1)*N] > tau) & (alpha_[i*N:(i + 1)*N] < (C * w_) - tau)) == True)[0]
                # Solve Least-Square problem to obtain the bias using the Support Vectros
                if len(SV_) > 0: b = y_tilde_[idx_[i*N:(i + 1)*N]][SV_] - K_tilde_[idx_[i*N:(i + 1)*N][SV_], :] @ alpha_ - epsilon
                else:            b = y_tilde_[idx_[i*N:(i + 1)*N]]      - K_tilde_[idx_[i*N:(i + 1)*N], :]      @ alpha_ - epsilon
                # Average bias and Support Vectors per output variable
                bias_[i, :] = b.mean()
                SVs_.append(SV_)
            # Return the average bias for each Support Vector
            return bias_, SVs_
        # Solve quadratic problem to obtain the dual paramters
        a_, a_star_ = ___dual(K_tilde_, K_hat_, Y_, w_, dXYZ_, std_, C, epsilon)
        # Solve Least-Square problem to obtain the bias using the Support Vectros
        b_, iSVs_ = ___bias(K_tilde_, Y_, w_, a_star_, C, epsilon, tau)
        return a_, a_star_, b_, iSVs_

    # Calculing prediction for new sample
    def __predict(K_tilde_, a_star_, b_):
        y_tilde_hat_ = K_tilde_.T @ a_star_
        N = y_tilde_hat_.shape[0]//2
        return np.concatenate((y_tilde_hat_[:N][:, np.newaxis], y_tilde_hat_[N:][:, np.newaxis]), axis = 1) + b_.T
    # Multi-output Kernels
    def __kernel(X_, Y_, kernel, degree, beta = None, gamma = None, independent = 0):
        # Implementation of Kernel Functions...
        def ___K(X, Y, beta, gamma, kernel, degree):
            if kernel is 'linear': return X @ Y.T
            if kernel is 'poly':   return (gamma*X @ Y.T + beta)**degree
            if kernel is 'rbf':    return np.exp(-gamma*euclidean_distances(X, Y)**2)
        # Autocorrelation Zero
        O_ = np.zeros((X_.shape[0], Y_.shape[0]))
        if independent == 1:
            x_ = X_[..., 0][:, np.newaxis]
            y_ = X_[..., 1][:, np.newaxis]
            x_star_ = Y_[..., 0][:, np.newaxis]
            y_star_ = Y_[..., 1][:, np.newaxis]
            K_11_ = ___K(x_, x_star_, beta, gamma, kernel, degree)
            #K_12_ = ___K(x_, y_star_, beta, gamma, kernel, degree)
            #K_21_ = ___K(y_, x_star_, beta, gamma, kernel, degree)
            K_22_ = ___K(y_, y_star_, beta, gamma, kernel, degree)
            K_tilde_ = np.concatenate((np.concatenate((K_11_, O_), axis = 0), np.concatenate((O_, K_22_), axis = 0)), axis = 1)
            #K_tilde_ = np.concatenate((np.concatenate((K_11_, -K_12_), axis = 0), np.concatenate((K_21_, K_22_), axis = 0)), axis = 1)
        else:
            # Kernel submatrix for Grand Matrix
            K_1_ = ___K(X_, Y_, beta, gamma, kernel, degree)
            K_2_ = ___K(X_, Y_, beta, gamma, kernel, degree)
            K_tilde_ = np.concatenate((np.concatenate((K_1_, O_), axis = 0), np.concatenate((O_, K_2_), axis = 0)), axis = 1)
        # Grand Matrix construct out of kernel submatrix and 0 matrix
        return K_tilde_

    # Flow Daynamic Constrain Operators Definition
    dV_, dD_ = _flow_dynamics_constrains_operator(dXYZ_, N_y, N_x)

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(Y_tr_[:, 0], Y_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(Y_tr_[:, 0], Y_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(Y_ts_[:, 0], Y_ts_[:, 1], _stdz_u, _stdz_v)
    Y_tr_ = np.concatenate((U_tr_[:, np.newaxis], V_tr_[:, np.newaxis],), axis  = 1)
    Y_ts_ = np.concatenate((U_ts_[:, np.newaxis], V_ts_[:, np.newaxis],), axis  = 1)

    # CV-parameters for the x,y-velocity components
    cv_, WMSE_val = __optimal_parameters(), 0.
    if CV:
        cv_, WMSE_val = __cross_validation(XY_stdz_tr_ , XY_stdz_, Y_tr_, w_tr_, dXYZ_, cv_)

    # Evaluate kernel function for training and testing
    K_tr_  = __kernel(XY_stdz_tr_, XY_stdz_tr_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    K_ts_  = __kernel(XY_stdz_tr_, XY_stdz_ts_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    K_hat_ = __kernel(XY_stdz_tr_,    XY_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])
    k_hat_ = __kernel(XY_stdz_tr_,    xy_stdz_, kernel, degree, gamma = cv_[2], beta = cv_[3])

    m_, s_, S_ = _standardization(N_y, N_x, n = K_hat_.shape[0], m_x = _stdz_u.mean_, s_x = _stdz_u.var_, m_y = _stdz_v.mean_, s_y = _stdz_v.var_)

    # Fit model for the x,y-velocity components
    a_, a_star_, b_, iSVs_ = __fit(K_tr_, K_hat_, Y_tr_, w_tr_, dXYZ_, std_ = [m_, s_, S_], C = cv_[0], epsilon = cv_[1], tau = 1e-6)
    # Error Analysis
    uv_tr_hat_ = __predict(K_tr_, a_star_, b_)
    uv_ts_hat_ = __predict(K_ts_, a_star_, b_)

    # Extrapolate Velocity Field to each pixel on a frame
    UV_hat_ = __predict(K_hat_, a_star_, b_)
    uv_hat_ = __predict(k_hat_, a_star_, b_)

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(UV_hat_[:, 0], UV_hat_[:, 1], _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(uv_hat_[:, 0], uv_hat_[:, 1], _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(uv_tr_hat_[:, 0], uv_tr_hat_[:, 1], _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(uv_ts_hat_[:, 0], uv_ts_hat_[:, 1], _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    # Reshape Prediction
    #UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(UV_hat_[..., 0], UV_hat_[..., 1], uv_hat_[..., 0], uv_hat_[..., 1], uv_tr_hat_[..., 0], uv_tr_hat_[..., 1], uv_ts_hat_[..., 0], uv_ts_hat_[..., 1], N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [cv_], WMSE_val

# Define Standardization in Matrix from
def _standardization(N_y, N_x, n, m_x, s_x, m_y, s_y):
    # Standardization Variables
    m_x_ = np.ones((N_y*N_x))*m_x
    m_y_ = np.ones((N_y*N_x))*m_y
    s_x_ = np.ones((N_y*N_x))*np.sqrt(s_x)
    s_y_ = np.ones((N_y*N_x))*np.sqrt(s_y)
    S_x_ = np.ones((n, N_y*N_x))*np.sqrt(s_x)
    S_y_ = np.ones((n, N_y*N_x))*np.sqrt(s_y)
    # For operators
    m_ = np.concatenate((m_x_, m_y_), axis = 0)
    s_ = np.concatenate((s_x_, s_y_), axis = 0)
    S_ = np.concatenate((S_x_, S_y_), axis = 1)
    return m_, s_, S_

# Define Fuid Dynamics Operatros for the constrains
def _flow_dynamics_constrains_operator(dXYZ_, N_y, N_x):
    def __spatial_differential_operator(N_x, N_y):
        N = N_y * N_x
        # Define x-axis differential Matrix
        dX_ = np.tril(np.ones(N), k = -1) - np.tril(np.ones(N), k = -2) - np.eye(N)
        # Define y-axis differential Matrix
        dY_ = np.tril(np.ones(N), k = -N_x) - np.tril(np.ones(N), k = -N_x - 1) - np.eye(N)
        # Set to 0 differential edges of the image
        idx_dX_ = np.arange(N_x, N + 1, N_x) - 1
        idx_dY_ = np.arange(N - N_x, N, 1)
        dX_[:, idx_dX_] = 0.
        dY_[:, idx_dY_] = 0.
        # Define Operator
        return sparse.csr_matrix(np.concatenate((np.concatenate((dX_, np.zeros((N, N))), axis = 0),
                                                 np.concatenate((np.zeros((N, N)), dY_), axis = 0)), axis = 1))
    # Define Divergence Operator
    def __divergence_operator(N_x, N_y):
        N = N_x * N_y
        dX_, dY_ = np.eye(N), np.eye(N)
        # Set to 0 differential edges of the image
        idx_dX_ = np.arange(0, N_x, 1)
        idx_dY_ = np.arange(0, N, N_x)
        dX_[:, idx_dX_] = 0.
        dY_[:, idx_dY_] = 0.
        # Define Vorticy operator
        return sparse.csr_matrix(np.concatenate((dX_, dY_), axis = 0))
    # Define Vorticy Operator
    def __vorticity_operator(N_x, N_y):
        N = N_x * N_y
        dX_, dY_ = np.eye(N), np.eye(N)
        # Set to 0 differential edges of the image
        idx_dX_ = np.arange(0, N_x, 1)
        idx_dY_ = np.arange(0, N, N_x)
        dX_[:, idx_dX_] = 0.
        dY_[:, idx_dY_] = 0.
        # Define Vorticy operator
        return sparse.csr_matrix(np.concatenate((dX_, -dY_), axis = 0))
    # Get Differetial operator
    dXdY_ = __spatial_differential_operator(N_x, N_y)
    # Get Vorticy operator
    Vor_ = __vorticity_operator(N_x, N_y)
    # Get Diveregence operator
    Div_ = __divergence_operator(N_x, N_y)
    # Differetial Operator and Vorticy Opterator
    dVor_ = dXdY_ @ Vor_
    # Differetial Operator and Divergence Opterator
    dDiv_ = dXdY_ @ Div_
    return dVor_ @ dVor_.T, dDiv_ @ dDiv_.T

# Subsample Dataset iff subsampling is True
def _subsample_dataset(X_, Y_, w_, n_subsample = 250, seed = 0, subsampling = True):
    if subsampling:
        np.random.seed()
        idx_ = np.random.permutation(np.arange(X_.shape[0], dtype = int))[:n_subsample]
        return X_[idx_, :], Y_[idx_, :], w_[idx_]
    else:
        return X_, Y_, w_

# How much divergence and vorticity has the flow estimation?
def _flow_dynamics_analysis(UV_hat_, dV_, dD_, verbose = True):
    # Calculating Vorticy Constrain
    def __vor(UV_hat_, dV_):
        Y_hat_ = np.concatenate((UV_hat_[..., 0].flatten(), UV_hat_[..., 1].flatten()), axis = 0)
        return Y_hat_.T @ dV_ @ Y_hat_
    # Calculating Divergence Constrain
    def __div(UV_hat_, dD_):
        Y_hat_ = np.concatenate((UV_hat_[..., 0].flatten(), UV_hat_[..., 1].flatten()), axis = 0)
        return Y_hat_.T @ dD_ @ Y_hat_
    D = __div(UV_hat_, dV_)
    V = __vor(UV_hat_, dD_)
    if verbose:
        print('>> Div.: {} Vor.: {}'.format(D, V))
    return D, V

# # How much divergence and vorticity has the flow estimation?
# def _flow_dynamics_analysis(V_x_, V_y_, dXYZ_, verbose = True):
#     #Div_ = _divergence(15*V_x_/dXYZ_[..., 0], 15*V_y_/dXYZ_[..., 1])
#     #Vor_ = _vorticity(15*V_x_/dXYZ_[..., 0], 15*V_y_/dXYZ_[..., 1])
#     Div_ = _divergence(V_x_, V_y_)
#     Vor_ = _vorticity(V_x_, V_y_)
#     # V_x_prime_ = 15. * V_x_/dXYZ_[..., 0]
#     # V_y_prime_ = 15. * V_y_/dXYZ_[..., 1]
#     # # Differentials in the x-axis
#     # dV_x_ = V_x_prime_[:, 1:] - V_x_prime_[:, :-1]
#     # # Differentials in the y-axis
#     # dV_y_ = V_y_prime_[1:, :] - V_y_prime_[:-1, :]
#     # # Compute Divergence and Vorticy
#     # Div_ = dV_x_[1:, :] + dV_y_[:, 1:]
#     # Vor_ = dV_x_[1:, :] - dV_y_[:, 1:]
#     # Compute total amount of Divergence and Vorticy in a frame
#     div = np.linalg.norm(Div_)
#     vor = np.linalg.norm(Vor_)
#     if verbose:
#         print('>> Div.: {} Vor.: {} Total: {}'.format(div, vor, np.sqrt(div**2 + vor**2)))
#     return div, vor


# Wind velocity estimation by using selected standardized vectors
def _wind_velocity_field_svm(XY_tr_, UV_tr_, w_tr_, XY_ts_, UV_ts_, w_ts_, dXYZ_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y,
                             N_y, N_x, step_size, svm, kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display):
    # How much error in the Regression?
    def __regression_error_analysis(UV_tr_, UV_ts_, UV_tr_hat_, UV_ts_hat_, w_tr_, w_ts_, verbose = True):
        def __MAPE(y_, y_hat_):
            return 100.*np.mean(np.absolute( (y_ - y_hat_) / y_ ) )
        def __WMAPE(y_, y_hat_, w_):
            return 100.*np.average(np.absolute( (y_ - y_hat_) / y_ ), weights = w_)
        # Calcualte multioutput weighted mean squared error
        MO_WRMSE_tr_ = np.sqrt(mean_squared_error(UV_tr_, UV_tr_hat_, sample_weight = w_tr_, multioutput = 'uniform_average'))
        MO_WRMSE_ts_ = np.sqrt(mean_squared_error(UV_ts_, UV_ts_hat_, sample_weight = w_ts_, multioutput = 'uniform_average'))
        # Calcualte multioutput weighted mean absolute error
        MO_WMAE_tr_ = mean_absolute_error(UV_tr_, UV_tr_hat_, sample_weight = w_tr_, multioutput = 'uniform_average')
        MO_WMAE_ts_ = mean_absolute_error(UV_ts_, UV_ts_hat_, sample_weight = w_ts_, multioutput = 'uniform_average')
        # Calcualte multioutput weighted mean absolute percentage error
        MO_WMAPE_tr_ = .5*__WMAPE(UV_tr_[:, 0], UV_tr_hat_[:, 0], w_tr_) + .5*__WMAPE(UV_tr_[:, 1], UV_tr_hat_[:, 1], w_tr_)
        MO_WMAPE_ts_ = .5*__WMAPE(UV_ts_[:, 0], UV_ts_hat_[:, 0], w_ts_) + .5*__WMAPE(UV_ts_[:, 1], UV_ts_hat_[:, 1], w_ts_)
        if verbose:
            print('>> WRMSE train: {} WRMSE test: {}'.format(MO_WRMSE_tr_, MO_WRMSE_ts_))
            print('>> WMAE train: {} WMAE test: {}'.format(MO_WMAE_tr_, MO_WMAE_ts_))
            print('>> WMAPE train: {} WMAPE test: {}'.format(MO_WMAPE_tr_, MO_WMAPE_ts_))
        # Calcualte multioutput mean squared error
        MO_RMSE_tr_ = np.sqrt(mean_squared_error(UV_tr_, UV_tr_hat_, multioutput = 'uniform_average'))
        MO_RMSE_ts_ = np.sqrt(mean_squared_error(UV_ts_, UV_ts_hat_, multioutput = 'uniform_average'))
        # Calcualte multioutput mean absolute error
        MO_MAE_tr_ = mean_absolute_error(UV_tr_, UV_tr_hat_, multioutput = 'uniform_average')
        MO_MAE_ts_ = mean_absolute_error(UV_ts_, UV_ts_hat_, multioutput = 'uniform_average')
        # Calcualte multioutput weighted mean absolute percentage error
        MO_MAPE_tr_ = .5*__MAPE(UV_tr_[:, 0], UV_tr_hat_[:, 0]) + .5*__MAPE(UV_tr_[:, 1], UV_tr_hat_[:, 1])
        MO_MAPE_ts_ = .5*__MAPE(UV_ts_[:, 0], UV_ts_hat_[:, 0]) + .5*__MAPE(UV_ts_[:, 1], UV_ts_hat_[:, 1])
        if verbose:
            print('>> RMSE train: {} RMSE test: {}'.format(MO_RMSE_tr_, MO_RMSE_ts_))
            print('>> MAE train: {} MAE test: {}'.format(MO_MAE_tr_, MO_MAE_ts_))
            print('>> MAPE train: {} MAPE test: {}'.format(MO_MAPE_tr_, MO_MAPE_ts_))
        return MO_WRMSE_tr_, MO_WRMSE_ts_, MO_RMSE_tr_, MO_RMSE_ts_, MO_WMAE_tr_, MO_WMAE_ts_, MO_MAE_tr_, MO_MAE_ts_

    # Applying standardization to the data and defining adequately the dataset prior to the SVM
    def __standardization(XY_train_, XY_test_, x_stdz, y_stdz):
        # Defining standardize covariate dataset
        XY_stdz_tr_ = np.concatenate((x_stdz.transform(XY_train_[..., 0][..., np.newaxis]),
                                      y_stdz.transform(XY_train_[..., 1][..., np.newaxis])), axis = 1)
        XY_stdz_ts_ = np.concatenate((x_stdz.transform( XY_test_[..., 0][..., np.newaxis]),
                                      y_stdz.transform( XY_test_[..., 1][..., np.newaxis])),  axis = 1)
        return XY_stdz_tr_, XY_stdz_ts_

    # Only Support Vector Machine here...
    def __eSVM(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, W_tr_, W_ts_, dXYZ_, N_y, N_x, step_size,
               svm, kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display):
        # Sklearn - Epsilon SVM for Regression

        if svm == 0:
            return _SK_eSVM_regression(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                                       kernel, degree, weights, N_grid, N_kfold, display)

        # Epsilon SVM for Regression
        if svm == 1:
            return _eSVM_regression(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                                    kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display)

        # Multi-output - Epsilon SVM for Regression
        if svm == 2:
            return _MO_eSVM_regression(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                                       kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display)

        # Multi-output with Flow Constrains - Epsilon SVM for Regression
        if svm == 3:
            return  _MO_eSVM_flow_constrains(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, dXYZ_, N_y, N_x, step_size,
                                             kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display)

    # Subsample the training dataset
    #XY_tr_, UV_tr_, w_tr_ = _subsample_dataset(XY_tr_, UV_tr_, w_tr_, n_subsample = 150, subsampling = True, seed = 0)

    # Standardarization of the vecotors selected coordinates for training an eSVM
    XY_tr_stdz_, XY_ts_stdz_ = __standardization(XY_tr_, XY_ts_, _stdz_x, _stdz_y)

    # I can use any other SVM here... However, input and output varibles has to remain the same
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, params_, WRMSE_val = __eSVM(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, dXYZ_, N_y, N_x, step_size,
                                                                            svm, kernel, degree, weights, CV, N_grid, N_kfold, N_BO_iterations, display)

    # Compute MO-Weigthed Error
    WRMSE_tr, WRMSE_ts, MRSE_tr, MRSE_ts, WMAE_tr, WMAE_ts, MAE_tr, MAE_ts = __regression_error_analysis(UV_tr_, UV_ts_, uv_tr_hat_, uv_ts_hat_, w_tr_, w_ts_, verbose = True)

    # Flow Daynamic Constrain Operators Definition
    dV_, dD_ = _flow_dynamics_constrains_operator(dXYZ_, N_y, N_x)

    # Compute Flow Dynamics Metrics
    D, V = _flow_dynamics_analysis(UV_hat_, dV_, dD_, verbose = True)
    #div, vor = _flow_dynamics_analysis(UV_hat_[..., 0], UV_hat_[..., 1], dXYZ_, verbose = True)
    return UV_hat_, [params_, WRMSE_val, WRMSE_ts, MRSE_ts, WMAE_ts, MAE_ts, D, V]

# Ridge Regression multiple output for regression optimized by SLSQP method
def _MO_RR(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, Y_tr_, Y_ts_, w_tr_, w_ts_, N_y, N_x,
           step_size, kernel, degree, weights, CV, N_grid, N_kfold, display  = None):
    # Grid Search optimization method
    def __cross_validation(X_tr_, Y_tr_, w_tr_, theta_):
        # Training a model for a cross-validation iteration
        def ___CV(X_tr_, X_val_, Y_tr_, Y_val_, w_tr_, w_val_, lamb, bias, beta, gamma):
            # Validataion kernel for training and test
            K_tr_  = __kernel(X_tr_,  X_tr_, kernel, degree, bias, gamma, beta)
            K_val_ = __kernel(X_tr_, X_val_, kernel, degree, bias, gamma, beta)
            # Fiting MO-RR model
            A_ = __fit(K_tr_, Y_tr_, w_tr_, lamb)
            Y_val_hat_ = np.nan_to_num(__predict(K_val_, A_))
            # Calculating the prediction error
            u_WMSE_val = mean_squared_error(Y_val_[:, 0], Y_val_hat_[:, 0],
                                            sample_weight = w_val_, multioutput = 'uniform_average')
            v_WMSE_val = mean_squared_error(Y_val_[:, 1], Y_val_hat_[:, 1],
                                            sample_weight = w_val_, multioutput = 'uniform_average')
            return .5*u_WMSE_val + .5*v_WMSE_val
        # Implementation of K-fold cross-validation
        def ___KF(arg_, X_tr_, Y_tr_, w_tr_):
            # Obtaining only values for the variables optimzied
            def ____interface(arg_, x_0 = 0, x_1 = 0, x_2 = 0, x_3 = 0):
                x_0 = 10**arg_[0]
                x_1 = 10**arg_[1]
                if len(arg_) > 2: x_2 = 10**arg_[2]
                if len(arg_) > 3: x_3 = 10**arg_[3]
                return x_0, x_1, x_2, x_3
            lamb, bias, gamma, beta = ____interface(np.squeeze(arg_))
            e_ = np.zeros((N_kfold))
            k  = 0
            for idx_tr_, idx_val_ in KFold(n_splits = N_kfold, random_state = None, shuffle = False).split(X_tr_):
                e_[k] = ___CV(X_tr_[idx_tr_, :], X_tr_[idx_val_, :], Y_tr_[idx_tr_],
                              Y_tr_[idx_val_], w_tr_[idx_tr_], w_tr_[idx_val_],
                              lamb, bias, beta, gamma)
                k += 1
            if np.isnan(e_.mean()):
                return 1e10
            else:
                return e_.mean()
        # Adapt parameters dictionary to GS optimization function
        def ___int_bounds(theta_, N_grid, bounds_ = ((1e-10, 1e2), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3))):
            log_bounds_ = []
            for i in range(len(theta_)):
                if theta_[i] != 0.0:
                    log_bounds_.append((np.log10(bounds_[i][0]), np.log10(bounds_[i][1])))
            return log_bounds_
        # Optima to logarithmic form
        def ___int_optima(opt_, x_ = [0., 0., 0., 0.]):
            for i in range(len(opt_)): x_[i] = 10**opt_[i]
            return x_
        # Optimization bounds defintion
        bounds_ = ___int_bounds(theta_, N_grid)
        # Grid-Search Optimization
        opt_ = optimize.brute(___KF, bounds_, args = (X_tr_, Y_tr_, w_tr_),
                              Ns = N_grid, full_output = True, finish = None)
        return ___int_optima(opt_[0]), opt_[1]
    # Saved optimal parameters
    def __optimal_parameters():
        if kernel is 'linear' and degree is 0: return [1., 1., 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [1., 1., 1., 0.]
        if kernel is 'poly'   and degree is 2: return [1., 1., 1., 1.]
        if kernel is 'poly'   and degree is 3: return [1., 1., 1., 1.]
    # Multi-output Ridge Regression model fit...
    def __fit(K_tilde_, Y_, w_, lamb):
        return np.linalg.pinv(K_tilde_ + np.eye(K_tilde_.shape[0])*lamb) @ Y_
    # Calculing prediction for new sample
    def __predict(K_tilde_, A_):
        return K_tilde_.T @ A_
    # Kernels
    def __kernel(X_, Y_, kernel, degree, bias, beta = None, gamma = None, independent = 0):
        # Implementation of Kernel Functions...
        def ___K(X, Y, bias, beta, gamma, kernel, degree):
            if kernel is 'linear': return X @ Y.T + bias
            if kernel is 'rbf':    return np.exp(-gamma*euclidean_distances(X, Y)**2) + bias
            if kernel is 'poly':   return (gamma*X @ Y.T + beta)**degree + bias
        # Grand Matrix construct out of kernel submatrix and 0 matrix
        return ___K(X_, Y_, bias, beta, gamma, kernel, degree)

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(Y_tr_[:, 0], Y_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(Y_tr_[:, 0], Y_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(Y_ts_[:, 0], Y_ts_[:, 1], _stdz_u, _stdz_v)
    Y_tr_ = np.concatenate((U_tr_[:, np.newaxis], V_tr_[:, np.newaxis],), axis  = 1)
    Y_ts_ = np.concatenate((U_ts_[:, np.newaxis], V_ts_[:, np.newaxis],), axis  = 1)

    # CV-parameters for x, y components
    cv_, WMSE_val = __optimal_parameters(), 0.
    if CV: cv_, WMSE_val = __cross_validation(XY_stdz_tr_, Y_tr_, w_tr_, cv_)
    print(cv_)
    # Evaluate kernel function for training and testing
    K_tr_  = __kernel(XY_stdz_tr_, XY_stdz_tr_, kernel, degree, bias = cv_[1], gamma = cv_[2], beta = cv_[3])
    K_ts_  = __kernel(XY_stdz_tr_, XY_stdz_ts_, kernel, degree, bias = cv_[1], gamma = cv_[2], beta = cv_[3])
    K_hat_ = __kernel(XY_stdz_tr_,    XY_stdz_, kernel, degree, bias = cv_[1], gamma = cv_[2], beta = cv_[3])
    k_hat_ = __kernel(XY_stdz_tr_,    xy_stdz_, kernel, degree, bias = cv_[1], gamma = cv_[2], beta = cv_[3])

    # Fit model for the x,y-velocity components
    A_ = __fit(K_tr_, Y_tr_, w_tr_, lamb = cv_[0])

    # Error Analysis
    uv_tr_hat_ = __predict(K_tr_, A_)
    uv_ts_hat_ = __predict(K_ts_, A_)
    # Extrapolate Velocity Field to each pixel on a frame
    UV_hat_ = __predict(K_hat_, A_)
    uv_hat_ = __predict(k_hat_, A_)

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(UV_hat_[:, 0], UV_hat_[:, 1], _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(uv_hat_[:, 0], uv_hat_[:, 1], _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(uv_tr_hat_[:, 0], uv_tr_hat_[:, 1], _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(uv_ts_hat_[:, 0], uv_ts_hat_[:, 1], _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    # Reshape Prediction
    #UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(UV_hat_[..., 0], UV_hat_[..., 1], uv_hat_[..., 0], uv_hat_[..., 1], uv_tr_hat_[..., 0], uv_tr_hat_[..., 1], uv_ts_hat_[..., 0], uv_ts_hat_[..., 1], N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [cv_], WMSE_val

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, degree):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        if kernel is 'linear': self.covar_module = gpytorch.kernels.LinearKernel()
        if kernel is 'rbf':    self.covar_module = gpytorch.kernels.RBFKernel()
        if kernel is 'poly':   self.covar_module = gpytorch.kernels.PolynomialKernel(power = degree)

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Gaussian Process Regression multiple output for regression optimized by SLSQP method
def _GPR(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, Y_tr_, Y_ts_, w_tr_, w_ts_, N_y, N_x,
           step_size, kernel, degree, weights, CV, N_grid, N_kfold, display  = None):
    # Saved optimal parameters
    def __optimal_parameters():
        if kernel is 'linear' and degree is 0: return [1., 1., 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [1., 1., 1., 0.]
        if kernel is 'poly'   and degree is 2: return [1., 1., 1., 1.]
        if kernel is 'poly'   and degree is 3: return [1., 1., 1., 1.]
    def __optimize(_model, _likel, X_, y_, w_, training_iter = 50):
        X_tr_ = torch.tensor(X_, dtype = torch.float)
        y_tr_ = torch.tensor(y_, dtype = torch.float)
        # Find optimal model hyperparameters
        _model.train()
        # Use the adam optimizer
        _optimizer = torch.optim.Adam(_model.parameters(), lr = 0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            _optimizer.zero_grad()
            # Output from model
            f_hat_tr_ = _model(X_tr_)
            # Calc loss and backprop gradients
            _error = - _mll(f_hat_tr_, y_tr_)
            _error.backward()
            print('Iter %d/%d - Loss: %.3f noise: %.3f' % (i + 1, training_iter, _error.item(), _model.likelihood.noise.item()))
            _optimizer.step()
        return _model, _likel
    # Gaussian Process Regression model fit...
    def __fit(X_, y_, w_, kernel, degree):
        X_tr_ = torch.tensor(X_, dtype = torch.float)
        y_tr_ = torch.tensor(y_, dtype = torch.float)
        #X_tr_ = torch.linspace(0, 1, 100)
        #y_tr_ = torch.sin(X_tr_ * (2 * math.pi)) + torch.randn(X_tr_.size()) * math.sqrt(0.04)
        # initialize likelihood and model
        _likel = gpytorch.likelihoods.GaussianLikelihood()
        _model = ExactGPModel(X_tr_, y_tr_, _likel, kernel, degree)
        return _model, _likel

    # Calculing prediction for new sample
    def __predict(_model, _likel, X_):
        X_ts_ = torch.tensor(X_, dtype = torch.float)
        _model.eval()
        _likel.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_hat_ = _likel(_model(X_ts_))
            return f_hat_.mean.numpy()

    # CV, Train, and predict SVM
    def __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, y_tr_, y_ts_, w_tr_, w_ts_,
                kernel, degree, weights, UV):
        # CV-parameters for the x-component
        cv_, WMSE_val = __optimal_parameters(), 0.
        #if CV: cv_, WMSE_val = __cross_validation(XY_stdz_tr_, y_tr_, w_tr_, cv_)
        # Fit model for the x,y-velocity components
        _model, _likel = __fit(XY_stdz_tr_, y_tr_, w_tr_, kernel, degree)
        _model, _likel = __optimize(_model, _likel, XY_stdz_tr_, y_tr_, w_tr_)
        # Error Analysis
        y_tr_hat_ = __predict(_model, _likel, XY_stdz_tr_)
        y_ts_hat_ = __predict(_model, _likel, XY_stdz_ts_)
        # Extrapolate Velocity Field to each pixel on a frame
        Y_hat_ = __predict(_model, _likel, XY_stdz_)
        y_hat_ = __predict(_model, _likel, xy_stdz_)
        return WMSE_val, cv_, Y_hat_, y_hat_, y_tr_hat_, y_ts_hat_


    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(Y_tr_[:, 0], Y_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(Y_tr_[:, 0], Y_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(Y_ts_[:, 0], Y_ts_[:, 1], _stdz_u, _stdz_v)

    #U_tr_, V_tr_ = Y_tr_[:, 0], Y_tr_[:, 1]
    #U_ts_, V_ts_ = Y_ts_[:, 0], Y_ts_[:, 1]

    # Train SVM for the U component of the velocity field
    u_WMSE_val, u_cv_, U_hat_, u_hat_, u_tr_hat_, u_ts_hat_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, U_tr_, U_ts_, w_tr_, w_ts_,
                                                                      kernel, degree, weights, UV = 'U')

    # Train SVM for the V component of the velocity field
    v_WMSE_val, v_cv_, V_hat_, v_hat_, v_tr_hat_, v_ts_hat_ = __train(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, V_tr_, V_ts_, w_tr_, w_ts_,
                                                                      kernel, degree, weights, UV = 'U')

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(U_hat_, V_hat_, _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(u_hat_, v_hat_, _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(u_tr_hat_, v_tr_hat_, _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(u_ts_hat_, v_ts_hat_, _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [u_cv_, v_cv_], .5*u_WMSE_val + .5*v_WMSE_val

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, degree, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks = num_tasks)
        if kernel is 'linear': self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.LinearKernel(), num_tasks = num_tasks, rank = 1)
        if kernel is 'rbf':    self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks = num_tasks, rank = 1)
        if kernel is 'poly':   self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.PolynomialKernel(power = degree), num_tasks = num_tasks, rank = 1)

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# Epsilon-SVM multiple output for regression optimized by SLSQP method
def _MO_GPR(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, Y_tr_, Y_ts_, w_tr_, w_ts_, N_y, N_x, step_size,
                        kernel, degree, weights, CV, N_grid, N_kfold, display  = None):

    # Saved optimal parameters
    def __optimal_parameters():
        # 31.06 0.31 0 0
        # 38.71 0.36 17.81 0
        # 34.73 0.28 4.65 10.66
        # 38.47 0.22 4.19 2.33
        if kernel is 'linear' and degree is 0: return [31.06, 0.31, 0., 0.]
        if kernel is 'rbf'    and degree is 0: return [8.71, 0.36, 17.81, 0.]
        if kernel is 'poly'   and degree is 2: return [34.73, 0.28, 4.65, 10.66]
        if kernel is 'poly'   and degree is 3: return [38.47, 0.22, 4.19, 2.33]

    def __optimize(_model, _likel, X_, Y_, w_, training_iter = 50):
        X_tr_ = torch.tensor(X_, dtype = torch.float)
        Y_tr_ = torch.tensor(Y_, dtype = torch.float)
        # Find optimal model hyperparameters
        _model.train()
        _likel.train()
        # Use the adam optimizer
        _optimizer = torch.optim.Adam([{'params': _model.parameters()},], lr = 0.1)
        # "Loss" for GPs - the marginal log likelihood
        _mll = gpytorch.mlls.ExactMarginalLogLikelihood(_likel, _model)
        for i in range(training_iter):
            _optimizer.zero_grad()
            _f_hat = _model(X_tr_)
            _error = - _mll(_f_hat, Y_tr_)
            _error.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, _error.item()))
            _optimizer.step()
        return _model, _likel
    # Gaussian Process Multi-ouput model fit...
    def __fit(X_, Y_, w_, kernel, degree, num_tasks):
        X_tr_ = torch.tensor(X_, dtype = torch.float)
        Y_tr_ = torch.tensor(Y_, dtype = torch.float)
        # initialize likelihood and models
        _likel = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = num_tasks)
        _model = MultitaskGPModel(X_tr_, Y_tr_, _likel, kernel, degree, num_tasks)
        return _model, _likel

    # Calculing prediction for new sample
    def __predict(_model, _likel, X_):
        X_ts_ = torch.tensor(X_, dtype = torch.float)
        # Set into eval mode
        _model.eval()
        _likel.eval()
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _f_hat = _likel(_model(X_ts_))
            return _f_hat.mean.numpy()

    # Fit parameters of the parameters of Training and test Dataset
    _stdz_u, _stdz_v = _wind_velocity_field_fit(Y_tr_[:, 0], Y_tr_[:, 1])

    # Transformation of Training and test Dataset
    U_tr_, V_tr_ = _wind_velocity_field_scale(Y_tr_[:, 0], Y_tr_[:, 1], _stdz_u, _stdz_v)
    U_ts_, V_ts_ = _wind_velocity_field_scale(Y_ts_[:, 0], Y_ts_[:, 1], _stdz_u, _stdz_v)
    Y_tr_ = np.concatenate((U_tr_[:, np.newaxis], V_tr_[:, np.newaxis],), axis  = 1)
    Y_ts_ = np.concatenate((U_ts_[:, np.newaxis], V_ts_[:, np.newaxis],), axis  = 1)

    # CV-parameters for x, y components
    cv_, WMSE_val = __optimal_parameters(), 0.
    print(cv_)
    # Fit model for the x,y-velocity components
    _model, _likel = __fit(XY_stdz_tr_, Y_tr_, w_tr_, kernel, degree, num_tasks = Y_tr_.shape[-1])
    _model, _likel  = __optimize(_model, _likel, XY_stdz_tr_, Y_tr_, w_tr_, training_iter = 50)

    # Error Analysis
    uv_tr_hat_ = __predict(_model, _likel, XY_stdz_tr_)
    uv_ts_hat_ = __predict(_model, _likel, XY_stdz_ts_)
    # Extrapolate Velocity Field to each pixel on a frame
    UV_hat_ = __predict(_model, _likel, XY_stdz_)
    uv_hat_ = __predict(_model, _likel, xy_stdz_)

    # Invert transformation of Prediction Dense and Spares
    U_hat_, V_hat_ = _wind_velocity_field_inverse(UV_hat_[:, 0], UV_hat_[:, 1], _stdz_u, _stdz_v)
    u_hat_, v_hat_ = _wind_velocity_field_inverse(uv_hat_[:, 0], uv_hat_[:, 1], _stdz_u, _stdz_v)

    # Invert transformation of Training and test Dataset
    u_tr_hat_, v_tr_hat_ = _wind_velocity_field_inverse(uv_tr_hat_[:, 0], uv_tr_hat_[:, 1], _stdz_u, _stdz_v)
    u_ts_hat_, v_ts_hat_ = _wind_velocity_field_inverse(uv_ts_hat_[:, 0], uv_ts_hat_[:, 1], _stdz_u, _stdz_v)

    # Reshape Prediction
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(U_hat_, V_hat_, u_hat_, v_hat_, u_tr_hat_, v_tr_hat_, u_ts_hat_, v_ts_hat_, N_y, N_x, step_size)

    # Reshape Prediction
    #UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_ = _reshape(UV_hat_[..., 0], UV_hat_[..., 1], uv_hat_[..., 0], uv_hat_[..., 1], uv_tr_hat_[..., 0], uv_tr_hat_[..., 1], uv_ts_hat_[..., 0], uv_ts_hat_[..., 1], N_y, N_x, step_size)

    return UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, [cv_], WMSE_val

# Wind velocity estimation by using selected standardized vectors
def _wind_velocity_field_gpr(XY_tr_, UV_tr_, w_tr_, XY_ts_, UV_ts_, w_ts_, dXYZ_, XY_stdz_, xy_stdz_, _stdz_x, _stdz_y,
                             N_y, N_x, step_size, gpr, kernel, degree, weights, CV, N_grid, N_kfold, display):
    # How much error in the Regression?
    def __regression_error_analysis(UV_tr_, UV_ts_, UV_tr_hat_, UV_ts_hat_, w_tr_, w_ts_, verbose = True):
        def __MAPE(y_, y_hat_):
            return 100.*np.mean(np.absolute( (y_ - y_hat_) / y_ ) )
        def __WMAPE(y_, y_hat_, w_):
            return 100.*np.average(np.absolute( (y_ - y_hat_) / y_ ), weights = w_)
        # Calcualte multioutput weighted mean squared error
        MO_WRMSE_tr_ = np.sqrt(mean_squared_error(UV_tr_, UV_tr_hat_, sample_weight = w_tr_, multioutput = 'uniform_average'))
        MO_WRMSE_ts_ = np.sqrt(mean_squared_error(UV_ts_, UV_ts_hat_, sample_weight = w_ts_, multioutput = 'uniform_average'))
        # Calcualte multioutput weighted mean absolute error
        MO_WMAE_tr_ = mean_absolute_error(UV_tr_, UV_tr_hat_, sample_weight = w_tr_, multioutput = 'uniform_average')
        MO_WMAE_ts_ = mean_absolute_error(UV_ts_, UV_ts_hat_, sample_weight = w_ts_, multioutput = 'uniform_average')
        # Calcualte multioutput weighted mean absolute percentage error
        MO_WMAPE_tr_ = .5*__WMAPE(UV_tr_[:, 0], UV_tr_hat_[:, 0], w_tr_) + .5*__WMAPE(UV_tr_[:, 1], UV_tr_hat_[:, 1], w_tr_)
        MO_WMAPE_ts_ = .5*__WMAPE(UV_ts_[:, 0], UV_ts_hat_[:, 0], w_ts_) + .5*__WMAPE(UV_ts_[:, 1], UV_ts_hat_[:, 1], w_ts_)
        if verbose:
            print('>> WRMSE train: {} WRMSE test: {}'.format(MO_WRMSE_tr_, MO_WRMSE_ts_))
            print('>> WMAE train: {} WMAE test: {}'.format(MO_WMAE_tr_, MO_WMAE_ts_))
            print('>> WMAPE train: {} WMAPE test: {}'.format(MO_WMAPE_tr_, MO_WMAPE_ts_))
        # Calcualte multioutput mean squared error
        MO_RMSE_tr_ = np.sqrt(mean_squared_error(UV_tr_, UV_tr_hat_, multioutput = 'uniform_average'))
        MO_RMSE_ts_ = np.sqrt(mean_squared_error(UV_ts_, UV_ts_hat_, multioutput = 'uniform_average'))
        # Calcualte multioutput mean absolute error
        MO_MAE_tr_ = mean_absolute_error(UV_tr_, UV_tr_hat_, multioutput = 'uniform_average')
        MO_MAE_ts_ = mean_absolute_error(UV_ts_, UV_ts_hat_, multioutput = 'uniform_average')
        # Calcualte multioutput weighted mean absolute percentage error
        MO_MAPE_tr_ = .5*__MAPE(UV_tr_[:, 0], UV_tr_hat_[:, 0]) + .5*__MAPE(UV_tr_[:, 1], UV_tr_hat_[:, 1])
        MO_MAPE_ts_ = .5*__MAPE(UV_ts_[:, 0], UV_ts_hat_[:, 0]) + .5*__MAPE(UV_ts_[:, 1], UV_ts_hat_[:, 1])
        if verbose:
            print('>> RMSE train: {} RMSE test: {}'.format(MO_RMSE_tr_, MO_RMSE_ts_))
            print('>> MAE train: {} MAE test: {}'.format(MO_MAE_tr_, MO_MAE_ts_))
            print('>> MAPE train: {} MAPE test: {}'.format(MO_MAPE_tr_, MO_MAPE_ts_))
        return MO_WRMSE_tr_, MO_WRMSE_ts_, MO_RMSE_tr_, MO_RMSE_ts_, MO_WMAE_tr_, MO_WMAE_ts_, MO_MAE_tr_, MO_MAE_ts_
    # Applying standardization to the data and defining adequately the dataset prior to the SVM
    def __standardization(XY_train_, XY_test_, x_stdz, y_stdz):
        # Defining standardize covariate dataset
        XY_stdz_tr_ = np.concatenate((x_stdz.transform(XY_train_[..., 0][..., np.newaxis]),
                                      y_stdz.transform(XY_train_[..., 1][..., np.newaxis])), axis = 1)
        XY_stdz_ts_ = np.concatenate((x_stdz.transform( XY_test_[..., 0][..., np.newaxis]),
                                      y_stdz.transform( XY_test_[..., 1][..., np.newaxis])),  axis = 1)
        return XY_stdz_tr_, XY_stdz_ts_

    # Only Guassian Process and Ridge Regression here...
    def __GPR(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, W_tr_, W_ts_, dXYZ_, N_y, N_x,
              step_size, gpr, kernel, degree, weights, CV, N_grid, N_kfold, display):
        # Gaussian Process for Regression
        if gpr == 0:
            return _GPR(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x,
                        step_size, kernel, degree, weights, N_grid, N_kfold, display)
        # Multi-output - Ridge Regression
        if gpr == 1:
            return _MO_RR(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x,
                          step_size, kernel, degree, weights, CV, N_grid, N_kfold, display)

        # Multi-output - Gaussian Process for Regression
        if gpr == 2:
            return _MO_GPR(XY_stdz_tr_, XY_stdz_ts_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_, w_tr_, w_ts_, N_y, N_x,
                           step_size, kernel, degree, weights, CV, N_grid, N_kfold, display)

    # Subsample the training dataset
    #XY_tr_, UV_tr_, w_tr_ = _subsample_dataset(XY_tr_, UV_tr_, w_tr_, n_subsample = 150, subsampling = True, seed = 0)

    # Standardarization of the vecotors selected coordinates for training an eSVM
    XY_tr_stdz_, XY_ts_stdz_ = __standardization(XY_tr_, XY_ts_, _stdz_x, _stdz_y)
    #XY_tr_stdz_ = XY_tr_
    #XY_ts_stdz_ = XY_ts_
    # I can use any other SVM here... However, input and output varibles has to remain the same
    UV_hat_, uv_hat_, uv_tr_hat_, uv_ts_hat_, params_, WRMSE_val = __GPR(XY_tr_stdz_, XY_ts_stdz_, XY_stdz_, xy_stdz_, UV_tr_, UV_ts_,
                                                                         w_tr_, w_ts_, dXYZ_, N_y, N_x, step_size, gpr, kernel, degree,
                                                                         weights, CV, N_grid, N_kfold, display)

    # Compute MO-Weigthed Error
    WRMSE_tr, WRMSE_ts, MRSE_tr, MRSE_ts, WMAE_tr, WMAE_ts, MAE_tr, MAE_ts = __regression_error_analysis(UV_tr_, UV_ts_, uv_tr_hat_,
                                                                                        uv_ts_hat_, w_tr_, w_ts_, verbose = True)

    # Flow Daynamic Constrain Operators Definition
    dV_, dD_ = _flow_dynamics_constrains_operator(dXYZ_, N_y, N_x)

    # Compute Flow Dynamics Metrics
    D, V = _flow_dynamics_analysis(UV_hat_, dV_, dD_, verbose = True)
    #div, vor = _flow_dynamics_analysis(UV_hat_[..., 0], UV_hat_[..., 1], dXYZ_, verbose = True)
    return UV_hat_, [params_, WRMSE_val, WRMSE_ts, MRSE_ts, WMAE_ts, MAE_ts, D, V]



__all__ = ['_wind_velocity_field_coordinates', '_wind_velocity_field_svm', '_wind_velocity_field_gpr']
