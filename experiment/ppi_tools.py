# from numerical_simulation import constraints
import numpy as np
import pandas as pd

import xgboost as xgb
import torch
import sklearn

from typing import Optional, Tuple, Union
from tqdm import tqdm

from scipy.optimize import minimize, least_squares, root_scalar
from scipy.special import expit

class Dataset:
    def __init__(self, X, y, predictors, proba=False):
        self.X = X
        self.y = y

        self.length = len(X)
        self.K = len(predictors)
        self.proba = proba
        self.predictors = predictors

        if self.proba:
            self.pesudo_y = [predictor.predict_proba(self.X) for predictor in self.predictors]
        else:
            self.pesudo_y = [predictor.predict(self.X) for predictor in self.predictors]
        self.pesudo_y = np.array(self.pesudo_y)


    def evaluate_inference(self, inference_func_, target_func, true_value, n, N, n_bootstrap=100, bootstrap_replace=True, **kwargs):
        """
        Evaluates inference methods using bootstrap resampling.
        """
        
        # --- 1. Validation & Setup ---
        if bootstrap_replace and (n + N > self.length):
            raise ValueError(
                f"Sample size (n={n} + N={N}) exceeds dataset length ({self.length}) "
                "with replacement=True. Reduce sizes or set replace=False."
            )

        # Pre-allocate arrays for results to improve performance
        # Shape: (n_bootstrap, )
        con_errors = np.zeros(n_bootstrap)
        con_est_vars = np.zeros(n_bootstrap)
        con_targets = np.zeros(n_bootstrap)
        
        # Shape: (n_bootstrap, K)
        ppi_errors = np.zeros((n_bootstrap, self.K))
        ppi_est_vars = np.zeros((n_bootstrap, self.K))
        ppi_est_vars_rectifier = np.zeros((n_bootstrap, self.K))
        ppi_targets = np.zeros((n_bootstrap, self.K))

        # Shape: (n_bootstrap, )
        ppimoe_errors = np.zeros(n_bootstrap)
        ppimoe_est_vars = np.zeros(n_bootstrap)
        ppimoe_est_vars_rectifier = np.zeros(n_bootstrap)
        ppimoe_targets = np.zeros(n_bootstrap)

        # Shape: (n_bootstrap, K) - assuming beta is size K
        moe_weights = np.zeros((n_bootstrap, self.K))

        # --- 2. Bootstrap Loop ---
        for i in tqdm(range(n_bootstrap)):
            # Sample indices
            indices = np.random.choice(self.length, size=n + N, replace=bootstrap_replace)
            idx_labeled = indices[:n]
            idx_unlabeled = indices[n:]

            # Prepare Labeled Data
            X_lbl = self.X[idx_labeled]
            y_lbl = self.y[idx_labeled]
            y_hat_lbl = self.pesudo_y[:, idx_labeled] # Shape (K, n)

            # Prepare Unlabeled Data
            X_unlbl = self.X[idx_unlabeled]
            y_hat_unlbl = self.pesudo_y[:, idx_unlabeled] # Shape (K, N)

            # A. Conventional Inference
            est, var = inference_func_(X=X_lbl, y=y_lbl, **kwargs)
            con_errors[i] = est - true_value
            con_est_vars[i] = var
            con_targets[i] = target_func(est, 100, self.X, self.y, n, N, ppi=None, **kwargs)

            # B. Single PPI (Loop over K predictors)
            for k in range(self.K):
                # Note: Passing specific k-th predictor row
                est, se_rectifier, var = inference_func_(
                    X=X_lbl, y=y_lbl, **kwargs,
                    y_hat=y_hat_lbl[k:k+1, :], # Keep dims 2D if needed, or [k]
                    tilde_X=X_unlbl, 
                    tilde_y_hat=y_hat_unlbl[k:k+1, :], 
                    ppi="single"
                )
                ppi_errors[i, k] = est - true_value
                ppi_est_vars_rectifier[i, k] = se_rectifier
                ppi_est_vars[i, k] = var
                ppi_targets[i, k] = target_func(est, 100, self.X, self.y, n, N, y_hat=self.pesudo_y[k], ppi="single", **kwargs)

            # C. PPI MoEs (Mixture of Experts)
            est, se_rectifier, var, weights = inference_func_(
                X=X_lbl, y=y_lbl, **kwargs,
                y_hat=y_hat_lbl, 
                tilde_X=X_unlbl, 
                tilde_y_hat=y_hat_unlbl, 
                ppi="MoEs"
            )
            ppimoe_errors[i] = est - true_value
            ppimoe_est_vars[i] = var
            ppimoe_est_vars_rectifier[i] = se_rectifier
            moe_weights[i] = weights

        for i in range(n_bootstrap):
            ppimoe_targets[i] = target_func(est, 100, self.X, self.y, n, N, y_hat=self.pesudo_y, weight=np.mean(moe_weights, axis=0), ppi="MoEs", **kwargs)


        # --- 3. Metrics Calculation ---
        
        # Helper to calculate metrics for a specific error array
        def calc_metrics(errors, estimated_vars, targets, estimated_vars_rectifier=None, **kwargs):
            n = len(errors)
            abs_e = np.abs(errors)
            sq_e  = errors ** 2
            centered_sq_e = (errors - np.mean(errors)) ** 2
            rect_vals = estimated_vars_rectifier
            return {
                "MAE":                    np.mean(abs_e),
                "MAE_se":                 np.std(abs_e, ddof=1) / np.sqrt(n),
                "err_var":                np.var(errors, ddof=1),
                "err_var_se":             np.std(centered_sq_e, ddof=1) / np.sqrt(n),
                "MSE":                    np.mean(sq_e),
                "MSE_se":                 np.std(sq_e, ddof=1) / np.sqrt(n),
                "est_var(Rectifier)":     np.mean(rect_vals) if rect_vals is not None else None,
                "est_var(Rectifier)_se":  np.std(rect_vals, ddof=1) / np.sqrt(n) if rect_vals is not None else None,
                "est_var(total)":         np.mean(estimated_vars),
                "est_var(total)_se":      np.std(estimated_vars, ddof=1) / np.sqrt(n),
                "target":                 np.mean(targets),
                "target_se":              np.std(targets, ddof=1) / np.sqrt(n),
            }

        # Conventional
        metrics_con = calc_metrics(con_errors, con_est_vars, con_targets, **kwargs)
        
        # PPI_MoEs
        metrics_moe = calc_metrics(ppimoe_errors, ppimoe_est_vars, ppimoe_targets, ppimoe_est_vars_rectifier, **kwargs)

        # PPI Single (Aggregated)
        # Calculate metrics per predictor first
        n_b = len(ppi_errors)
        abs_ppi_e = np.abs(ppi_errors)                                      # (n_b, K)
        sq_ppi_e  = ppi_errors ** 2                                         # (n_b, K)
        ctr_ppi_e = (ppi_errors - np.mean(ppi_errors, axis=0)) ** 2        # (n_b, K)
        ppi_mae_k                = np.mean(abs_ppi_e, axis=0)
        ppi_mae_se_k             = np.std(abs_ppi_e, axis=0, ddof=1) / np.sqrt(n_b)
        ppi_var_k                = np.var(ppi_errors, axis=0, ddof=1)
        ppi_var_se_k             = np.std(ctr_ppi_e, axis=0, ddof=1) / np.sqrt(n_b)
        ppi_mse_k                = np.mean(sq_ppi_e, axis=0)
        ppi_mse_se_k             = np.std(sq_ppi_e, axis=0, ddof=1) / np.sqrt(n_b)
        ppi_est_vars_k           = np.mean(ppi_est_vars, axis=0)
        ppi_est_vars_se_k        = np.std(ppi_est_vars, axis=0, ddof=1) / np.sqrt(n_b)
        ppi_est_vars_rectifier_k = np.mean(ppi_est_vars_rectifier, axis=0)
        ppi_est_vars_rect_se_k   = np.std(ppi_est_vars_rectifier, axis=0, ddof=1) / np.sqrt(n_b)
        ppi_targets_k            = np.mean(ppi_targets, axis=0)
        ppi_targets_se_k         = np.std(ppi_targets, axis=0, ddof=1) / np.sqrt(n_b)

        best_idx = ppi_targets_k.argmin()
        worst_idx = ppi_targets_k.argmax()

        def _ppi_k_dict(k):
            return {
                "MAE": ppi_mae_k[k], "MAE_se": ppi_mae_se_k[k],
                "err_var": ppi_var_k[k], "err_var_se": ppi_var_se_k[k],
                "MSE": ppi_mse_k[k], "MSE_se": ppi_mse_se_k[k],
                "est_var(Rectifier)": ppi_est_vars_rectifier_k[k], "est_var(Rectifier)_se": ppi_est_vars_rect_se_k[k],
                "est_var(total)": ppi_est_vars_k[k], "est_var(total)_se": ppi_est_vars_se_k[k],
                "target": ppi_targets_k[k], "target_se": ppi_targets_se_k[k],
            }

        # --- 4. Construct DataFrame ---
        data = {
            "Conventional": metrics_con,
            "PPI_best": _ppi_k_dict(best_idx),
            "PPI_mean": {
                "MAE": ppi_mae_k.mean(), "MAE_se": ppi_mae_se_k.mean(),
                "err_var": ppi_var_k.mean(), "err_var_se": ppi_var_se_k.mean(),
                "MSE": ppi_mse_k.mean(), "MSE_se": ppi_mse_se_k.mean(),
                "est_var(Rectifier)": ppi_est_vars_rectifier_k.mean(), "est_var(Rectifier)_se": ppi_est_vars_rect_se_k.mean(),
                "est_var(total)": ppi_est_vars_k.mean(), "est_var(total)_se": ppi_est_vars_se_k.mean(),
                "target": ppi_targets_k.mean(), "target_se": ppi_targets_se_k.mean(),
            },
            "PPI_worst": _ppi_k_dict(worst_idx),
            "PPI_MoEs": metrics_moe
        }

        # Transpose to match desired format (Rows: Methods, Cols: Metrics)
        evaluation_df = pd.DataFrame(data).T 
        
        # Reorder columns to match original output preference
        evaluation_df = evaluation_df[["MAE", "MAE_se", "err_var", "err_var_se", "MSE", "MSE_se",
                                        "est_var(Rectifier)", "est_var(Rectifier)_se",
                                        "est_var(total)", "est_var(total)_se", "target", "target_se"]]

        return evaluation_df, np.mean(moe_weights, axis=0)



    def evaluate_inference_multi(self, inference_func_, target_func, true_value, n, N, n_bootstrap=100, bootstrap_replace=True, n_informative=None, **kwargs):
        """
        Evaluates multivariate inference methods using bootstrap resampling.
        
        Args:
            inference_func_: Function returning (est, cov) or (est, cov_rect, cov_total, weights).
            true_value: The ground truth vector of shape (d,).
            n: Labeled sample size.
            N: Unlabeled sample size.
        """
        
        # --- 0. Determine Dimension ---
        # Ensure true_value is an array
        true_value = np.array(true_value)
        if true_value.ndim == 0:
            d = 1
            true_value = true_value.reshape(1)
        else:
            d = true_value.shape[0]

        # --- 1. Validation & Setup ---
        if bootstrap_replace and (n + N > self.length):
            raise ValueError(
                f"Sample size (n={n} + N={N}) exceeds dataset length ({self.length}) "
                "with replacement=True. Reduce sizes or set replace=False."
            )

        # --- 2. Pre-allocate arrays ---
        # Errors: (n_bootstrap, d)
        # Covariances: (n_bootstrap, d, d)
        
        # Conventional
        con_errors = np.zeros((n_bootstrap, d))
        con_est_covs = np.zeros((n_bootstrap, d, d))
        con_targets = np.zeros((n_bootstrap, d, d))

        # PPI Single: (n_bootstrap, K, d) for errors, (n_bootstrap, K, d, d) for covs
        ppi_errors = np.zeros((n_bootstrap, self.K, d))
        ppi_est_covs = np.zeros((n_bootstrap, self.K, d, d))
        ppi_est_covs_rectifier = np.zeros((n_bootstrap, self.K, d, d))
        ppi_targets_k = np.zeros((n_bootstrap, self.K, d, d))

        # PPI MoE
        ppimoe_errors = np.zeros((n_bootstrap, d))
        ppimoe_est_covs = np.zeros((n_bootstrap, d, d))
        ppimoe_est_covs_rectifier = np.zeros((n_bootstrap, d, d))
        ppimoe_targets = np.zeros((n_bootstrap, d, d))
        
        # Weights: (n_bootstrap, K)
        moe_weights = np.zeros((n_bootstrap, self.K))


        # --- 3. Bootstrap Loop ---
        for i in tqdm(range(n_bootstrap)):
            # Sample indices
            indices = np.random.choice(self.length, size=n + N, replace=bootstrap_replace)
            idx_labeled = indices[:n]
            idx_unlabeled = indices[n:]

            # Prepare Labeled Data
            X_lbl = self.X[idx_labeled]
            y_lbl = self.y[idx_labeled] # Shape (n, d)
            # print(np.mean(y_lbl), np.std(y_lbl))
            
            # Pseudo labels: Shape (K, n, d) or (K, n) depending on your implementation
            # Assuming self.pesudo_y is (K, Total_Samples, d) for multivariate
            y_hat_lbl = self.pesudo_y[:, idx_labeled] 

            # Prepare Unlabeled Data
            X_unlbl = self.X[idx_unlabeled]
            y_hat_unlbl = self.pesudo_y[:, idx_unlabeled]

            # A. Conventional Inference
            # Expects return: est (d,), cov (d, d)
            est, cov = inference_func_(X=X_lbl, y=y_lbl, **kwargs)
            
            if n_informative is not None:
                est = est[:n_informative]
                cov = cov[:n_informative, :n_informative]
                target = target_func(est, 100, X=self.X[:, :n_informative], y=self.y, n=n, N=N, ppi=None, **kwargs)
            else:
                target = target_func(est, 100, X=self.X, y=self.y, n=n, N=N, ppi=None, **kwargs)
            con_errors[i] = est - true_value
            con_est_covs[i] = cov
            con_targets[i] = target

            # B. Single PPI (Loop over K predictors)
            for k in range(self.K):
                # Expects return: est (d,), cov_rect (d, d), cov_total (d, d)
                est, cov_rect, cov_total = inference_func_(
                    X=X_lbl, y=y_lbl,
                    y_hat=y_hat_lbl[k:k+1].flatten(), # Keep dimension for predictor index
                    tilde_X=X_unlbl, 
                    tilde_y_hat=y_hat_unlbl[k:k+1].flatten(), 
                    ppi="single",
                    **kwargs
                )
                
                if n_informative is not None:
                    est = est[:n_informative]
                    cov_rect = cov_rect[:n_informative, :n_informative]
                    cov_total = cov_total[:n_informative, :n_informative]
                    target = target_func(est, 100, self.X[:, :n_informative], self.y, n, N, y_hat=self.pesudo_y[k], ppi="single", **kwargs)
                else:
                    target = target_func(est, 100, self.X, self.y, n, N, y_hat=self.pesudo_y[k], ppi="single", **kwargs)
                ppi_errors[i, k] = est - true_value
                ppi_est_covs_rectifier[i, k] = cov_rect
                ppi_est_covs[i, k] = cov_total
                ppi_targets_k[i, k] = target

            # C. PPI MoEs (Mixture of Experts)
            # Expects return: est (d,), cov_rect (d, d), cov_total (d, d), weights (K,)
            est, cov_rect, cov_total, weights = inference_func_(
                X=X_lbl, y=y_lbl,
                y_hat=y_hat_lbl, 
                tilde_X=X_unlbl, 
                tilde_y_hat=y_hat_unlbl, 
                ppi="MoEs",
                **kwargs
            )
            if n_informative is not None:
                est = est[:n_informative]
                cov_rect = cov_rect[:n_informative, :n_informative]
                cov_total = cov_total[:n_informative, :n_informative]
            #     target = target_func(est, 100, self.X[:, :n_informative], self.y, n, N, y_hat=self.pesudo_y, weight=weights, ppi="MoEs", **kwargs)
            # else:
            #     target = target_func(est, 100, self.X, self.y, n, N, y_hat=self.pesudo_y, weight=weights, ppi="MoEs", **kwargs)
            ppimoe_errors[i] = est - true_value
            ppimoe_est_covs[i] = cov_total
            ppimoe_est_covs_rectifier[i] = cov_rect
            moe_weights[i] = weights
            # ppimoe_targets[i] = target
        
        for i in range(n_bootstrap):
            if n_informative is not None:
                target = target_func(est, 100, self.X[:, :n_informative], self.y, n, N, y_hat=self.pesudo_y, weight=np.mean(moe_weights, axis=0), ppi="MoEs", **kwargs)
            else:
                target = target_func(est, 100, self.X, self.y, n, N, y_hat=self.pesudo_y, weight=np.mean(moe_weights, axis=0), ppi="MoEs", **kwargs)
            ppimoe_targets[i] = target
        
        
        # return con_targets, ppi_targets_k, ppimoe_targets

        # --- 4. Metrics Calculation Helper ---
        def calc_metrics(errors, estimated_covs, targets, estimated_covs_rectifier=None):
            """
            errors: (n_bootstrap, d)
            estimated_covs: (n_bootstrap, d, d)
            """
            n_b = errors.shape[0]
            # Per-trial scalar summaries for SE computation
            per_trial_sum_abs_e = np.sum(np.abs(errors), axis=1)              # (n_b,)
            per_trial_sum_sq_e  = np.sum(errors**2, axis=1)                   # (n_b,)
            per_trial_trace_tot = np.einsum('nii->n', estimated_covs)         # (n_b,)
            per_trial_target_tr = np.einsum('nii->n', targets)                # (n_b,)
            if estimated_covs_rectifier is not None:
                per_trial_trace_rect = np.einsum('nii->n', estimated_covs_rectifier)  # (n_b,)

            # 1. MAE (Vector of length d)
            mae = np.mean(np.abs(errors), axis=0)

            # 2. MSE (Vector of length d)
            mse = np.mean(errors**2, axis=0)

            # 3. err_cov -> Empirical Covariance Matrix of the errors (d, d)
            err_cov = np.cov(errors, rowvar=False)
            if d == 1: err_cov = np.array([[err_cov]])

            # 4. est_cov(total) -> Average of Estimated Covariance Matrices (d, d)
            avg_est_cov = np.mean(estimated_covs, axis=0)

            # 5. est_cov(Rectifier) -> Average of Estimated Rectifier Covariances (d, d)
            if estimated_covs_rectifier is not None:
                avg_est_cov_rect = np.mean(estimated_covs_rectifier, axis=0)
            else:
                avg_est_cov_rect = np.nan * np.ones((d, d))
            return {
                "Sum_MAE":                     np.sum(mae),
                "Sum_MAE_se":                  np.std(per_trial_sum_abs_e, ddof=1) / np.sqrt(n_b),
                "Trace_err_cov":               np.trace(err_cov),
                "Trace_err_cov_se":            np.std(per_trial_sum_sq_e, ddof=1) / np.sqrt(n_b),
                "Sum_MSE":                     np.sum(mse),
                "Sum_MSE_se":                  np.std(per_trial_sum_sq_e, ddof=1) / np.sqrt(n_b),
                "Trace_est_cov(Rectifier)":    np.trace(avg_est_cov_rect),
                "Trace_est_cov(Rectifier)_se": np.std(per_trial_trace_rect, ddof=1) / np.sqrt(n_b) if estimated_covs_rectifier is not None else None,
                "Trace_est_cov(Total)":        np.trace(avg_est_cov),
                "Trace_est_cov(Total)_se":     np.std(per_trial_trace_tot, ddof=1) / np.sqrt(n_b),
                "target":                      np.trace(np.mean(targets, axis=0)),
                "target_se":                   np.std(per_trial_target_tr, ddof=1) / np.sqrt(n_b),
            }


        
        # --- 5. Compute Metrics ---
    

        # Conventional
        metrics_con = calc_metrics(con_errors, con_est_covs, con_targets)
        
        # PPI_MoEs
        metrics_moe = calc_metrics(ppimoe_errors, ppimoe_est_covs, ppimoe_targets, ppimoe_est_covs_rectifier)

        # PPI Single (Aggregated)
        # We need to select Best/Worst/Mean based on a scalar criterion.
        # Common criterion for multivariate: Trace of the MSE matrix (sum of MSEs).
        
        # Calculate MSE per predictor: Shape (K, d)
        
        ppi_target_per_k = np.mean(ppi_targets_k, axis=0)
        # trace of K d x d matrices
        ppi_target_scalar_score = [np.trace(x) for x in ppi_target_per_k]

        
        
        best_k_idx = np.argmin(ppi_target_scalar_score)
        worst_k_idx = np.argmax(ppi_target_scalar_score)


        # print(ppi_targets_k.shape, ppi_target_scalar_score.shape, ppi_target_per_k.shape, best_k_idx, worst_k_idx)
        # Extract data for Best/Worst/Mean
        
        # Best
        metrics_ppi_best = calc_metrics(
            ppi_errors[:, best_k_idx, :], 
            ppi_est_covs[:, best_k_idx, :, :],
            ppi_targets_k[:, best_k_idx, :], 
            ppi_est_covs_rectifier[:, best_k_idx, :, :]
        )
        
        # Worst
        metrics_ppi_worst = calc_metrics(
            ppi_errors[:, worst_k_idx, :], 
            ppi_est_covs[:, worst_k_idx, :, :],
            ppi_targets_k[:, worst_k_idx, :], 
            ppi_est_covs_rectifier[:, worst_k_idx, :, :]
        )
        
        # Mean (Averaging the raw errors/covs across K first, then calc metrics? 
        # Or averaging the metrics? Usually averaging metrics is safer for reporting "Mean Performance")
        # Here we calculate metrics for all K, then average the resulting dictionaries.
        
        all_ppi_metrics = []
        for k in range(self.K):
            all_ppi_metrics.append(calc_metrics(
                ppi_errors[:, k, :], 
                ppi_est_covs[:, k, :, :],
                ppi_targets_k[:, k, :], 
                ppi_est_covs_rectifier[:, k, :, :]
            ))
        
        # Helper to average dictionaries of numpy arrays
        def avg_dicts(dict_list):
            avg_d = {}
            for key in dict_list[0].keys():
                if dict_list[0][key] is None:
                    avg_d[key] = None
                else:
                    avg_d[key] = np.mean([d[key] for d in dict_list], axis=0)
            return avg_d

        metrics_ppi_mean = avg_dicts(all_ppi_metrics)

        # --- 6. Construct DataFrame ---
        data = {
            "Conventional": metrics_con,
            "PPI_best": metrics_ppi_best,
            "PPI_mean": metrics_ppi_mean,
            "PPI_worst": metrics_ppi_worst,
            "PPI_MoEs": metrics_moe
        }

        
        # Transpose to match desired format
        evaluation_df = pd.DataFrame(data).T 
        
        # Reorder columns
        evaluation_df = evaluation_df[["Sum_MAE", "Sum_MAE_se", "Trace_err_cov", "Trace_err_cov_se",
                                        "Sum_MSE", "Sum_MSE_se", "Trace_est_cov(Rectifier)", "Trace_est_cov(Rectifier)_se",
                                        "Trace_est_cov(Total)", "Trace_est_cov(Total)_se", "target", "target_se"]]

        return evaluation_df, np.mean(moe_weights, axis=0)

def mean_value_inference(
    X: np.ndarray, 
    y: np.ndarray, 
    y_hat: Optional[np.ndarray] = None, 
    tilde_X: Optional[np.ndarray] = None, 
    tilde_y_hat: Optional[np.ndarray] = None, 
    ppi: Optional[str] = None,
    **kwargs
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    Estimates the population mean of y with optional Prediction-Powered Inference (PPI) corrections.

    Parameters:
    -----------
    X : np.ndarray
        Labeled features, shape (n, p).
    y : np.ndarray
        Labeled targets, shape (n, ).
    y_hat : np.ndarray, optional
        Predictions on labeled data, shape (K, n). Required for 'single' and 'MoEs'.
    tilde_X : np.ndarray, optional
        Unlabeled features, shape (N, p). Required for 'single' and 'MoEs'.
    tilde_y_hat : np.ndarray, optional
        Predictions on unlabeled data, shape (K, N). Required for 'single' and 'MoEs'.
    ppi : str, optional
        Inference method. Options: None (classical), 'single', 'MoEs'.
    level : float, default=0.95
        Confidence level (currently unused in calculation, but reserved for interval construction).

    Returns:
    --------
    If ppi is None:
        (mean_estimate, variance)
    If ppi is 'single':
        (mean_estimate, rectifier_variance, total_variance)
    If ppi is 'MoEs':
        (mean_estimate, rectifier_variance, total_variance, beta_coefficients)
    """
    
    n = X.shape[0]

    # --- 1. Classical Inference (No PPI) ---
    if ppi is None:
        mean_est = np.mean(y)
        var = np.var(y, ddof=1) / n
        return mean_est, var

    # Ensure required inputs are present for PPI methods
    if tilde_X is None or y_hat is None or tilde_y_hat is None:
        raise ValueError(f"y_hat, tilde_X, and tilde_y_hat are required for ppi='{ppi}'")

    N = tilde_X.shape[0]

    # --- 2. Single Predictor PPI ---
    if ppi == "single":
        # PPI
        # Estimate = Mean(Unlabeled Preds) - Mean(Labeled Preds) + Mean(Labeled True)
        mean_est = np.mean(tilde_y_hat) - np.mean(y_hat) + np.mean(y)
        
        # Standard Error calculation (Conservative sum of SEs)
        var_labeled = np.var(y - y_hat, ddof=1)
        var_unlabeled = np.var(tilde_y_hat, ddof=1)
        var_estimated = var_labeled / n + var_unlabeled / N
        
        return mean_est, var_labeled / n, var_estimated

    # --- 3. Mixture of Experts (MoEs) PPI ---
    elif ppi == "MoEs":
        # Shapes: y (n,), F (K, n), tilde_F (K, N)
        # Ensure y is centered for covariance calculation
        y_centered = y - np.mean(y)
        
        # F corresponds to predictions on labeled set (centered)
        F = y_hat
        F_mean = np.mean(F, axis=1, keepdims=True)
        F_centered = F - F_mean
        tilde_F_mean = np.mean(tilde_y_hat, axis=1, keepdims=True)
        tilde_F_centered = tilde_y_hat - tilde_F_mean

        # Calculate Covariance Matrix of Predictions (K x K)
        # equivalent to: F @ F.T / n - bar_F @ bar_F.T
        Sigma_F = np.cov(F, bias=True) 
        Sigma_tilde_F = np.cov(tilde_y_hat, bias=True)

        # Calculate Covariance between Predictions and Label (K, )
        # equivalent to: F @ y / n - bar_F * bar_y
        # We use dot product on centered data for numerical stability
        cov_Fy = (F_centered @ y_centered) / n

        # Solve for Beta: Sigma_F * beta = cov_Fy
        # Using linalg.solve is more stable than inv(Sigma_F) @ cov_Fy
        # Adding a small jitter (1e-6) to diagonal for regularization if matrix is singular
        try:
            hat_beta = np.linalg.solve(Sigma_F + n / N * Sigma_tilde_F, cov_Fy)
        except np.linalg.LinAlgError:
            hat_beta = np.linalg.lstsq(Sigma_F + n / N * Sigma_tilde_F, cov_Fy, rcond=None)[0]

        # Calculate Rectified Estimator
        # hat_theta = Mean(tilde_F.T @ beta) - (Mean(F.T @ beta) - Mean(y))
        
        # Predictions weighted by Beta
        preds_labeled_weighted = F.T @ hat_beta       # (n,)
        preds_unlabeled_weighted = tilde_y_hat.T @ hat_beta # (N,)

        # The bias correction term (delta)
        hat_delta_f = np.mean(preds_labeled_weighted) - np.mean(y)
        
        # Final Estimate
        hat_theta = np.mean(preds_unlabeled_weighted) - hat_delta_f

        # Standard Error Calculation
        rectifier = preds_labeled_weighted - y - hat_delta_f
        
        var_rectifier = np.var(rectifier, ddof=1)
        var_unlabeled = np.var(preds_unlabeled_weighted, ddof=1)
        
        hat_sigma = var_rectifier / n + var_unlabeled / N

        return hat_theta, var_rectifier / n, hat_sigma, hat_beta

    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")

def mean_value_target(theta, bootstrap, X, y, n, N=None, y_hat=None, weight=None, ppi=None, **kwargs):
    length = X.shape[0]
    record = np.zeros(bootstrap)
    if ppi is None:
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n, replace=True)
            y_lbl = y[indices]
            record[i] = np.mean(y_lbl)
        return np.var(record, ddof=1)
    elif ppi == "single":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            y_lbl = y[indices[:n]]
            y_hat_lbl = y_hat[indices[:n]]      
            tilde_y_hat = y_hat[indices[n:]]
            record[i] = np.mean(tilde_y_hat) - np.mean(y_hat_lbl) + np.mean(y_lbl)
        return np.var(record, ddof=1)
    elif ppi == "MoEs":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            y_lbl = y[indices[:n]]
            y_hat_lbl = weight @ y_hat[:, indices[:n]]      
            tilde_y_hat = weight @ y_hat[:, indices[n:]]
            record[i] = np.mean(tilde_y_hat) - np.mean(y_hat_lbl) + np.mean(y_lbl)
        return np.var(record, ddof=1)
    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")


def quantile_inference(q=None, X=None, y=None, y_hat=None, tilde_X=None, tilde_y_hat=None, ppi=None, epsilon=1e-6, h=0.1, **kwargs):
    """
    Estimates the quantile value of y with optional Prediction-Powered Inference (PPI) corrections.

    Parameters:
    -----------
    q : float
        Quantile value, between 0 and 1.
    X : np.ndarray
        Labeled features, shape (n, p).
    y : np.ndarray
        Labeled targets, shape (n, ).
    n : int
        Labeled sample size.
    N : int, optional
        Unlabeled sample size.
    y_hat : np.ndarray, optional
        Predictions on labeled data, shape (K, n). Required for 'single' and 'MoEs'.
    weight : np.ndarray, optional
        Weights for the MoEs, shape (K, ). Required for 'MoEs'.
    ppi : str, optional
        Inference method. Options: None (classical), 'single', 'MoEs'.

    Returns:
    --------
    If ppi is None:
        (quantile_estimate, variance)
    If ppi is 'single':
        (quantile_estimate, rectifier_variance, total_variance)
    If ppi is 'MoEs':
        (quantile_estimate, rectifier_variance, total_variance, beta_coefficients)
    """
    n = X.shape[0]
    # S is a plain sigmoid; h is the true bandwidth parameter (NOT cancelled).
    # S((theta - z) / h) approximates the CDF P(Z <= theta) as h -> 0.
    def S(x):
        return 1 / (1 + np.exp(-x))
    def S_prime(x):
        return S(x) * (1 - S(x))
    def S_prime_prime(x):
        return S(x) * (1 - S(x)) * (1 - 2 * S(x))

    def _kde_density(theta, arr):
        """Estimate f(theta) via KDE with Silverman bandwidth for numerical stability.

        When h << std(arr), the logistic kernel S' is numerically zero for nearly all
        observations, so we fall back to Silverman-bandwidth KDE to estimate f(theta).
        """
        arr_flat = np.asarray(arr).flatten()
        n_arr = len(arr_flat)
        h_silverman = 1.06 * np.std(arr_flat) * (n_arr ** (-0.2))
        h_eff = max(h, h_silverman)  # use Silverman if it dominates h
        return max(np.mean(S_prime((theta - arr_flat) / h_eff)) / h_eff, 0.0)

    if ppi is None:
        def f(theta):
            return np.mean(S((theta - y) / h)) - q
        res = root_scalar(f, bracket=[np.min(y), np.max(y)], method='brentq')
        theta = res.root
        # Delta method: Avar(θ̂) = Var(ψ)/n / (∂E[ψ]/∂θ)²  where ψ=S((θ-Y)/h)
        # Use KDE with adaptive bandwidth so denom ≈ f(θ) is numerically stable.
        denom = _kde_density(theta, y)
        return theta, np.var(S((theta - y) / h), ddof=1) / n / (denom ** 2 + epsilon)
    N = tilde_X.shape[0]

    if ppi == "single":

        def f(theta):
            # CDF convention: S((theta - z)/h) ~ P(Z <= theta)
            # PPI estimating equation: F_hat_unlbl + (F_y - F_hat_lbl) = q
            return np.mean(S((theta - tilde_y_hat) / h)) - np.mean(S((theta - y_hat) / h)) + np.mean(S((theta - y) / h)) - q

        res = root_scalar(f, bracket=[np.min(y), np.max(y)], method='brentq')
        theta = res.root

        # Variance of numerator terms
        var_rectifier = np.var(S((theta - y) / h) - S((theta - y_hat) / h), ddof=1)
        var_unlabeled = np.var(S((theta - tilde_y_hat) / h), ddof=1)
        var_estimated = var_rectifier / n + var_unlabeled / N

        # Delta method denominator: derivative of full estimating eq w.r.t. θ
        # Each term estimates the density of the respective distribution at θ.
        denom = (_kde_density(theta, tilde_y_hat)
                 - _kde_density(theta, y_hat)
                 + _kde_density(theta, y))
        denom2 = denom ** 2 + epsilon

        return theta, var_rectifier / n / denom2, var_estimated / denom2

    elif ppi == "MoEs":
        K = y_hat.shape[0]
        # Use Silverman bandwidth for the optimization objective so that SLSQP
        # receives non-zero gradients even when h << std(y).  The estimating
        # equation f(theta, weight) still uses h so that the returned theta is
        # the correct smooth-quantile estimator; only the variance objective
        # used during weight search is evaluated at h_opt.
        h_opt = max(h, 1.06 * np.std(y) * (n ** (-0.2)))

        def f(theta, weight):
            return np.mean(S((theta - weight @ tilde_y_hat) / h)) - np.mean(S((theta - weight @ y_hat) / h)) + np.mean(S((theta - y) / h)) - q

        def objective(weight):
            try:
                theta = root_scalar(f, bracket=[np.min(y), np.max(y)], method='brentq', args=(weight,)).root
            except ValueError:
                return np.inf
            var_rectifier = np.var(S((theta - y) / h_opt) - S((theta - weight @ y_hat) / h_opt), ddof=1)
            var_unlabeled = np.var(S((theta - weight @ tilde_y_hat) / h_opt), ddof=1)
            return var_rectifier + var_unlabeled / N * n

        res = minimize(
            objective,
            x0=np.ones(K) / K,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * K,
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
        )
        weight = res.x
        theta = root_scalar(f, bracket=[np.min(y), np.max(y)], method='brentq', args=(weight,)).root
        var_rectifier = np.var(S((theta - y) / h) - S((theta - weight @ y_hat) / h), ddof=1)
        var_unlabeled = np.var(S((theta - weight @ tilde_y_hat) / h), ddof=1)
        var_estimated = var_rectifier / n + var_unlabeled / N

        # Delta method denominator with adaptive KDE for each term
        denom = (_kde_density(theta, weight @ tilde_y_hat)
                 - _kde_density(theta, weight @ y_hat)
                 + _kde_density(theta, y))
        denom2 = denom ** 2 + epsilon

        return theta, var_rectifier / n / denom2, var_estimated / denom2, weight



    elif ppi == "MoEs_old_indicator":
        K = y_hat.shape[0]

        scale_factor = 1 #np.std(y) + 1e-8
        center_factor = 0 #np.mean(y)
        
        y_scaled = (y - center_factor) / scale_factor
        y_hat_scaled = (y_hat - center_factor) / scale_factor
        tilde_y_hat_scaled = (tilde_y_hat - center_factor) / scale_factor

        # scale_factor = 1
        # center_factor = 0
        # y_scaled = y + 0.0
        # y_hat_scaled = y_hat + 0.0
        # tilde_y_hat_scaled = tilde_y_hat + 0.0
        
        # 初始 theta 估计 (在缩放空间)
        theta_init_scaled = np.quantile(y_scaled, q, method="nearest")

        # --- 2. 辅助函数定义 ---

        def get_theta_constraint(w, current_theta_guess):
            """
            给定权重 w，寻找满足 PPI 约束的 theta。
            约束: mean(w @ tilde_y_hat < theta) - mean(w @ y_hat < theta) + mean(y < theta) - q = 0
            """
            def objective(t):
                # 预测值 (K, n) -> (n,)
                preds_U = w @ tilde_y_hat_scaled
                preds_L = w @ y_hat_scaled
                
                # 这里的逻辑对应 zero_gradient_MoEs
                g_theta = (preds_U < t).mean()
                delta = (preds_L < t).mean() - (y_scaled < t).mean()
                return (g_theta - delta - q) ** 2
                
            # 使用 Nelder-Mead 或 simple bounds 求解标量 theta
            # res = minimize(objective, x0=current_theta_guess, method='Nelder-Mead', tol=1e-5)
            res = minimize(objective,
                           x0=current_theta_guess, 
                           method='BFGS')
            return res.x[0]

        def smooth_loss_and_grad(w, theta_scaled, k=10.0):
            """
            计算平滑方差 Loss 和 Gradient
            w: (K,)
            y_hat_scaled: (K, n)
            """
            # --- 前向传播 ---
            # 组合预测值
            preds_L = w @ y_hat_scaled        # (n,)
            preds_U = w @ tilde_y_hat_scaled  # (N,)

            # Soft Indicators: Sigmoid(k * (theta - pred))
            # 注意: Indicator 是 I(pred < theta) <=> I(theta - pred > 0)
            arg_L = k * (theta_scaled - preds_L)
            arg_U = k * (theta_scaled - preds_U)
            
            S_L = expit(arg_L) # (n,)
            S_U = expit(arg_U) # (N,)

            # 真实标签 Indicator (对 y 不平滑，因为它是常数)
            I_y = (y_scaled < theta_scaled).astype(float)

            # 构造变量 Z
            Z_L = I_y - S_L
            Z_U = S_U

            # 中心化 (Centered variables)
            Z_L_centered = Z_L - np.mean(Z_L)
            Z_U_centered = Z_U - np.mean(Z_U)

            # --- Loss Calculation (Variance) ---
            # Var(Z) = sum((Z - mean)^2) / (n-1)
            var_L = np.sum(Z_L_centered ** 2) / (n - 1)
            var_U = np.sum(Z_U_centered ** 2) / (N - 1)
            
            # 最终 Loss: Var_L/n + Var_U/N
            loss = var_L / n + var_U / N

            # --- Gradient Calculation ---
            # 链式法则:
            # d(Loss)/dw = d(Loss)/d(Var) * d(Var)/d(Z) * d(Z)/d(S) * d(S)/d(arg) * d(arg)/dw
            
            # 1. d(S)/d(arg) * d(arg)/dw
            # d(arg)/dw = d(k*theta - k*w@X)/dw = -k * X
            # G_vec = S * (1-S) * (-k)
            
            # G_L_vec: (n,)
            G_L_vec = (S_L * (1 - S_L)) * (-k) 
            # G_U_vec: (N,)
            G_U_vec = (S_U * (1 - S_U)) * (-k)

            # 2. d(Var)/dw (结合前面所有项)
            # d(Var(Z))/dw = 2/(n-1) * sum( (Z - meanZ) * dZ/dw )
            # dZ_L/dw = - dS_L/dw (因为 Z_L = I_y - S_L)
            # dZ_U/dw = + dS_U/dw (因为 Z_U = S_U)
            
            # 梯度项: (Z - mean) * G_vec
            # shape: (n,) * (n,) = (n,)
            term_L = Z_L_centered * G_L_vec * (-1) # (-1 来自 dZ_L/dS_L)
            term_U = Z_U_centered * G_U_vec * (1)
            
            # 矩阵乘法聚合梯度: X @ term
            # (K, n) @ (n, 1) -> (K, 1)
            grad_L_component = y_hat_scaled @ term_L
            grad_U_component = tilde_y_hat_scaled @ term_U
            
            # 加上系数
            # Loss = Var/n => dLoss = 1/n * dVar
            # dVar = 2/(n-1) * ...
            # 总系数 = 2 / (n * (n-1))
            
            grad_L = (2 / (n * (n - 1))) * grad_L_component
            grad_U = (2 / (N * (N - 1))) * grad_U_component
            
            return loss, grad_L + grad_U

        # --- 3. 初始化策略 (Warm Start) ---
        # 这一步解决 "结果比 e_k 差" 的问题
        # 我们先计算所有单模型(One-hot)的 Variance，取最好的作为起点
        
        best_loss = float('inf')
        best_w = np.ones(K) / K # 默认均匀
        
        # 候选权重列表：均匀权重 + K个单模型
        candidates = [np.ones(K) / K] + [np.eye(K)[i] for i in range(K)]
        
        for w_cand in candidates:
            # 对应的 theta
            t_cand = get_theta_constraint(w_cand, theta_init_scaled)
            # 计算 Loss
            l_cand, _ = smooth_loss_and_grad(w_cand, t_cand, k=10.0)
            
            if l_cand < best_loss:
                best_loss = l_cand
                best_w = w_cand
                theta_init_scaled = t_cand # 更新 theta 猜测值

        # --- 4. 优化循环 ---
        weight = best_w
        theta = theta_init_scaled
        k_smooth = 10.0 # 标准化后，k=10 既平滑又有梯度

        for i in range(100): # 迭代次数
            prev_weight = weight.copy()
            
            # Step A: 更新 theta (满足约束)
            theta = get_theta_constraint(weight, theta)
            
            # Step B: 更新 weight (最小化方差)
            res = minimize(
                fun=smooth_loss_and_grad,
                x0=weight,
                args=(theta, k_smooth),
                method='SLSQP',
                # method='BFGS',
                jac=True,
                constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}),
                bounds=[(0, 1) for _ in range(K)],
                options={'ftol': 1e-12}
            )
            weight = res.x
            
            # 检查收敛
            if np.linalg.norm(weight - prev_weight) < epsilon:
                break
        # --- 5. 结果还原 ---
        theta = get_theta_constraint(weight, np.quantile(weight @ tilde_y_hat, q, method="nearest"))
        # 将 theta 还原回原始尺度
        final_theta = theta * scale_factor + center_factor
        # 计算最终的真实方差 (Hard Indicator)
        # Rectifier
        ind_y = (y < final_theta) * 1
        ind_yhat = (weight @ y_hat < final_theta) * 1
        var_rectifier = np.var(ind_y - ind_yhat, ddof=1) / n
        
        # Unlabeled
        ind_unlab = (weight @ tilde_y_hat < final_theta) * 1
        var_unlabeled = np.var(ind_unlab, ddof=1) / N
        
        var_estimated = var_rectifier + var_unlabeled
        
        return final_theta, var_rectifier, var_estimated, weight
    elif ppi == "MoEs_old":
        K = y_hat.shape[0]
        def zero_gradient_MoEs(theta, q, y, y_hat, tilde_y_hat, weight):
            g_theta = (weight @ tilde_y_hat < theta).mean()
            delta_theta = - (y < theta).mean() + (weight @ y_hat < theta).mean()
            return (g_theta - delta_theta - q) ** 2

        def variance_MoEs(weight, theta, q, y, y_hat, tilde_y_hat):
            n, N = y.shape[0], tilde_y_hat.shape[0]
            var_rectifier = np.var((y < theta) * 1 - (weight @ y_hat < theta) * 1)
            var_unlabeled = np.var((weight @ tilde_y_hat < theta) * 1)
            # S_L = (y < theta) * 1 - (weight @ y_hat < theta) * 1
            # S_U = (weight @ tilde_y_hat < theta) * 1
            
            # Z_L = S_L - np.mean(S_L)
            # Z_U = S_U - np.mean(S_U)

            # var_rectifier = np.mean(Z_L ** 2)
            # var_unlabeled = np.mean(Z_U ** 2)

            return var_rectifier / n  + var_unlabeled / N

        def variance_MoEs_smooth(w, theta, q, y, y_hat, tilde_y_hat, k=100.0):
            n, N = y.shape[0], tilde_y_hat.shape[0]
            # print(y[0], y_hat[:, 0], w, (w @ y_hat) [0])
            S_L = expit(k * (theta - w @ y_hat))       # soft indicator for Labeled 
            S_U = expit(k * (theta - w @ tilde_y_hat)) # soft indicator for Unlabeled 

            Z_L = (y < theta) - S_L
            Z_U = S_U
            diff_L = Z_L - np.mean(Z_L) # centered Z_L
            diff_U = Z_U - np.mean(Z_U) # centered Z_U

            loss = np.var(Z_L, ddof=1)/n + np.var(Z_U, ddof=1)/N

            G_L = k * S_L * (1 - S_L) # gradient of S_L
            G_U = k * S_U * (1 - S_U) # gradient of S_U

            grad_L = (2 / (n * (n - 1))) * (y_hat @ (diff_L * G_L))
            grad_U = (2 / (N * (N - 1))) * (tilde_y_hat @ (diff_U * -G_U))

            return loss, grad_L + grad_U
        
        weight = np.random.rand(K)
        weight = weight / np.sum(weight)

        for _ in range(1000):
            prev_weight = weight + 0.0
            theta = minimize(
                fun=zero_gradient_MoEs,
                x0=np.quantile(y, q, method="nearest"),
                args=(q, y, y_hat, tilde_y_hat, weight),
                method='SLSQP'
            ).x[0]
            
            weight = minimize(
                fun=variance_MoEs_smooth,
                x0=weight,
                args=(theta, q, y, y_hat, tilde_y_hat),
                method='SLSQP',
                jac=True,
                options={'eps': 1e-3},
                constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}),
                bounds=tuple((0, 1) for _ in range(K))
            ).x
            # print(_, theta, weight, y[0])

            # print(f'{variance_MoEs(weight, theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(weight, theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')
            
            # print(f'{variance_MoEs(np.zeros(K)+0.0, theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(np.zeros(K), theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')
            # print(f'{variance_MoEs(np.zeros(K)+0.002, theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(np.zeros(K)+0.002, theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')
            # print(f'{variance_MoEs(np.zeros(K)+0.005, theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(np.zeros(K)+0.005, theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')
            
            # print(f'{variance_MoEs(np.ones(K) / K, theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(np.ones(K) / K, theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')

            # print(f'{variance_MoEs(np.array([0, 1, 0, 0, 0, 0]), theta, q, y, y_hat, tilde_y_hat):.6f}, {variance_MoEs_smooth(np.array([0, 1, 0, 0, 0, 0]), theta, q, y, y_hat, tilde_y_hat)[0]:.6f}')
            
            if np.linalg.norm(weight - prev_weight) < epsilon:
                break
        var_rectifier = np.var((y < theta) * 1 - (weight @ y_hat < theta) * 1, ddof=1) / n
        var_unlabeled = np.var((weight @ tilde_y_hat < theta) * 1, ddof=1) / N
        var_estimated = var_rectifier + var_unlabeled
        return theta, var_rectifier, var_estimated, weight
    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")

def quantile_target(theta, bootstrap, X, y, n, N=None, y_hat=None, weight=None, ppi=None, q=None, h=0.1, **kwargs):
    def S(x):
        return 1 / (1 + np.exp(-x))
    length = X.shape[0]
    record = np.zeros(bootstrap)
    if ppi is None:
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n, replace=True)
            _y = y[indices]
            record[i] = np.mean(S((theta - _y) / h))
        return np.var(record, ddof=1)
    else:
        if ppi == "MoEs":
            y_hat = weight @ y_hat
        if ppi == "MoEs" or ppi == "single":
            for i in range(bootstrap):
                indices = np.random.choice(length, size=n+N, replace=True)
                _y = y[indices[:n]]
                _y_hat = y_hat[indices[:n]]
                _tilde_y_hat = y_hat[indices[n:]]

                g_theta = np.mean(S((theta - _tilde_y_hat) / h))
                delta_theta = - np.mean(S((theta - _y) / h)) + np.mean(S((theta - _y_hat) / h))
                record[i] = g_theta - delta_theta - q
            return np.var(record, ddof=1)
        else:
            raise ValueError(f"Unknown ppi mode: {ppi}")


def LR_coef_inference(
    X: np.ndarray, 
    y: np.ndarray, 
    y_hat: Optional[np.ndarray] = None, 
    tilde_X: Optional[np.ndarray] = None, 
    tilde_y_hat: Optional[np.ndarray] = None, 
    ppi: Optional[str] = None,
    moe_constraint: Optional[list[str]] = None,
    **kwargs
) -> Union[
    Tuple[np.ndarray, np.ndarray], 
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Estimates the Linear Regression Coefficient with optional Prediction-Powered Inference (PPI) corrections.

    Parameters:
    -----------
    X : np.ndarray
        Labeled features, shape (n, p).
    y : np.ndarray
        Labeled targets, shape (n,).
    y_hat : np.ndarray, optional
        Predictions on labeled data. 
        Shape (n,) for 'single'. Shape (K, n) for 'MoEs'.
    tilde_X : np.ndarray, optional
        Unlabeled features, shape (N, p).
    tilde_y_hat : np.ndarray, optional
        Predictions on unlabeled data.
        Shape (N,) for 'single'. Shape (K, N) for 'MoEs'.
    ppi : str, optional
        Inference method. Options: None (classical), 'single', 'MoEs'.

    Returns:
    --------
    If ppi is None:
        (est_coef, est_covariance)
    If ppi is 'single':
        (est_coef, rectifier_variance, total_variance)
    If ppi is 'MoEs':
        (est_coef, rectifier_variance, total_variance, beta_weights)
    """
    
    # --- 1. Input Standardization ---
    n, p = X.shape
    y = y.flatten()
    
    # Pre-compute Gram matrix inverse (using pinv for stability against collinearity)
    # hat_Sigma_inv = (X^T X / n)^-1
    XtX_inv_n = np.linalg.pinv(X.T @ X / n)
    
    # --- 2. Classical Inference (No PPI) ---
    if ppi is None:
        # OLS Estimator: (X^T X)^-1 X^T y
        est_coef = XtX_inv_n @ (X.T @ y / n)
        
        # Sandwich Variance Estimator (Huber-White / HC0)
        # Residuals: (n, 1)
        residuals = (y - X @ est_coef)[:, np.newaxis]
        # Score: x_i * e_i -> (n, p)
        scores = residuals * X 
        # M = 1/n * \sum (x_i e_i)(x_i e_i)^T
        M = scores.T @ scores / n
        
        est_coef_cov = XtX_inv_n @ M @ XtX_inv_n
        return est_coef, est_coef_cov / n 

    # --- 3. PPI Pre-requisites ---
    if tilde_X is None or y_hat is None or tilde_y_hat is None:
        raise ValueError(f"y_hat, tilde_X, and tilde_y_hat are required for ppi='{ppi}'")

    N = tilde_X.shape[0]
    # hat_tilde_Sigma_inv = (tilde_X^T tilde_X / N)^-1
    tilde_XtX_inv_N = np.linalg.pinv(tilde_X.T @ tilde_X / N)

    # --- 4. Single Predictor PPI ---
    if ppi == "single":
        # Ensure 1D shapes for single predictor
        y_hat = y_hat.flatten()         # (n,)
        tilde_y_hat = tilde_y_hat.flatten() # (N,)

        # 1. Estimate on Unlabeled (tilde_theta)
        # (tilde_X^T tilde_X)^-1 tilde_X^T tilde_y_hat
        tilde_theta = tilde_XtX_inv_N @ (tilde_X.T @ tilde_y_hat / N)

        # 2. Calculate Rectifier (delta_f) on Labeled
        # (X^T X)^-1 X^T (y_hat - y)
        delta_f = XtX_inv_n @ (X.T @ (y_hat - y) / n)

        # 3. Final Estimator
        hat_theta = tilde_theta - delta_f

        # 4. Variance Estimation
        # Variance from unlabeled data
        res_unlabeled = (tilde_y_hat - tilde_X @ tilde_theta)[:, np.newaxis]
        scores_unlabeled = res_unlabeled * tilde_X # (N, p)
        tilde_M = scores_unlabeled.T @ scores_unlabeled / N
        var_unlabeled = tilde_XtX_inv_N @ tilde_M @ tilde_XtX_inv_N

        # Variance from rectifier (labeled data)
        res_labeled = (y_hat - y - X @ delta_f)[:, np.newaxis]
        scores_labeled = res_labeled * X # (n, p)
        M = scores_labeled.T @ scores_labeled / n
        var_rectifier = XtX_inv_n @ M @ XtX_inv_n

        # Total Variance
        hat_sigma = (var_rectifier / n) + (var_unlabeled / N)

        return hat_theta, var_rectifier / n, hat_sigma 

    # --- 5. Mixture of Experts (MoEs) PPI ---
    elif ppi == "MoEs":
        # Ensure correct shapes: Experts should be (K, n)
        if y_hat.ndim == 1: y_hat = y_hat.reshape(1, -1)
        if tilde_y_hat.ndim == 1: tilde_y_hat = tilde_y_hat.reshape(1, -1)
        
        K = y_hat.shape[0]

        # Helper to compute H matrices (Covariance of gradients)
        def compute_H_matrix(residuals_K, Features, Sigma_inv):
            """
            Computes H matrix efficiently.
            residuals_K: (K, n_samples)
            Features: (n_samples, p)
            Sigma_inv: (p, p)
            Returns: (K, K) matrix
            """
            # Project features into parameter space: (n, p)
            proj_features = Features @ Sigma_inv 
            
            # Compute gradients per sample per expert: (n, K, p)
            # This is effectively: gradient_i_k = residual_{k,i} * (Sigma^-1 x_i)
            # We use einsum for clarity and memory efficiency
            # 'kn,np->nkp' maps (K, n) and (n, p) to (n, K, p)
            gradients = np.einsum('kn,np->nkp', residuals_K, proj_features)
            
            # Compute outer product sum: \sum_i grad_i grad_i^T
            # Result is (K, K, p, p) if we kept full covariance, 
            # but here we are solving for beta (weights), so we contract p.
            # The logic in the original code implies H is (K, K).
            # H_{a,b} = Mean_over_n [ (grad_a)^T (grad_b) ] ?? 
            # Re-analyzing original code: matrix_squares = einsum... -> nka. mean(axis=0) -> ka.
            # It computes the inner product of the gradients in p-space.
            
            # (n, K, p) @ (n, K, p) -> (n, K, K) via dot product on p
            H_per_sample = np.einsum('nkp,nap->nka', gradients, gradients)
            return np.mean(H_per_sample, axis=0)

        # --- Step A: Compute Regression Auxiliaries ---
        # Regress every expert k onto features X to get hat_A (p, K)
        hat_A = XtX_inv_n @ (X.T @ y_hat.T / n)
        
        # Regress y onto features X to get hat_b (p,)
        hat_b = XtX_inv_n @ (X.T @ y / n)
        
        # Regress unlabeled experts onto tilde_X to get hat_tilde_A (p, K)
        hat_tilde_A = tilde_XtX_inv_N @ (tilde_X.T @ tilde_y_hat.T / N)

        # --- Step B: Compute Variance Components (H matrices) ---
        # Residuals of experts on labeled data: (K, n)
        res_y_hat = y_hat - hat_A.T @ X.T 
        hat_H = compute_H_matrix(res_y_hat, X, XtX_inv_n)

        # Residuals of experts on unlabeled data: (K, N)
        res_tilde_y_hat = tilde_y_hat - hat_tilde_A.T @ tilde_X.T
        hat_tilde_H = compute_H_matrix(res_tilde_y_hat, tilde_X, tilde_XtX_inv_N)

        # --- Step C: Compute Target Vector r ---
        # We need Cross-Covariance between expert gradients and true target gradients
        # res_y: (1, n)
        res_y = (y - hat_b.T @ X.T).reshape(1, -1)
        
        # Re-use compute_H logic but with one side being the target residual
        proj_X = X @ XtX_inv_n
        grad_experts = np.einsum('kn,np->nkp', res_y_hat, proj_X) # (n, K, p)
        grad_target = np.einsum('an,np->nap', res_y, proj_X)      # (n, 1, p)
        
        # (n, K, p) dot (n, 1, p) -> (n, K, 1) -> mean -> (K, 1)
        hat_r = np.mean(np.einsum('nkp,nap->nka', grad_experts, grad_target), axis=0)

        
        # --- Step D: Solve for Beta (Optimal Expert Weights) ---
        # Regularization can be added to diagonal if singular, e.g., + 1e-6 * I
        if moe_constraint is None:
            hat_beta = np.linalg.solve(hat_H + (n/N) * hat_tilde_H, hat_r).flatten()
        else:
            from scipy.optimize import minimize, LinearConstraint, Bounds
            sH = hat_H + (n / N) * hat_tilde_H
            sH = np.asarray(sH, float)

            sr = np.asarray(hat_r, float).reshape(-1)  # force (K,)
            K = sH.shape[0]

            # sanity checks (建议保留，能立刻定位 shape 问题)
            if sH.shape != (K, K):
                raise ValueError(f"sH shape must be (K,K), got {sH.shape}")
            if sr.shape != (K,):
                raise ValueError(f"sr shape must be (K,), got {sr.shape}")

            def fun(x):
                x = np.asarray(x, float).reshape(-1)   # force (K,)
                return float(x @ (sH @ x) - 2.0 * (sr @ x))

            def grad(x):
                x = np.asarray(x, float).reshape(-1)
                return (sH + sH.T) @ x - 2.0 * sr

            def hess(x):
                return (sH + sH.T)

            cons = []
            if moe_constraint is not None and ("sum_to_one" in moe_constraint):
                cons.append(LinearConstraint(np.ones((1, K)), 1.0, 1.0))

            bounds = Bounds(0.0, np.inf) if (moe_constraint is not None and ("nonneg" in moe_constraint)) else None

            # x0 must be 1D (K,)
            try:
                x0 = np.linalg.solve(sH, sr)
            except np.linalg.LinAlgError:
                x0 = np.full(K, 1.0 / K) if ("sum_to_one" in (moe_constraint or [])) else np.zeros(K)
            x0 = np.asarray(x0, float).reshape(-1)

            result = minimize(
                fun,
                x0=x0,
                method="trust-constr",
                jac=grad,
                hess=hess,
                constraints=cons,     # must be list, not None
                bounds=bounds,
                options={"maxiter": 1000},
            )
            if not result.success:
                raise RuntimeError(result.message)

            hat_beta = result.x

            

        # --- Step E: Final Inference ---
        # 1. Unlabeled Estimate (Weighted combination of experts)
        # hat_theta_f = (tilde_X^T tilde_X)^-1 tilde_X^T (tilde_y_hat^T beta)
        hat_theta_f = tilde_XtX_inv_N @ (tilde_X.T @ (tilde_y_hat.T @ hat_beta) / N)

        # 2. Rectifier Estimate
        # hat_theta_delta = (X^T X)^-1 X^T (y_hat^T beta - y)
        hat_theta_delta = XtX_inv_n @ (X.T @ (y_hat.T @ hat_beta - y) / n)

        hat_theta = hat_theta_f - hat_theta_delta

        # --- Step F: Variance Estimation ---
        # Labeled Variance Component (W)
        # Residual: (y_hat^T beta - y) - X(A beta - b)
        # Note: (hat_A @ hat_beta) is the regression of the combined expert on X
        # print(y_hat.shape, hat_beta.shape, y.shape, hat_A.shape, hat_b.shape, X.shape)
        res_W = (y_hat.T @ hat_beta - y) - X @ (hat_A @ hat_beta - hat_b) # (n,)
        grad_W = XtX_inv_n @ (res_W[:, np.newaxis] * X).T # (p, n)
        hat_W = grad_W @ grad_W.T / n

        # Unlabeled Variance Component (W_tilde)
        # Residual: tilde_y_hat^T beta - tilde_X(tilde_A beta)
        res_W_tilde = (tilde_y_hat.T @ hat_beta) - (tilde_X @ (hat_tilde_A @ hat_beta)) # (N,)
        grad_W_tilde = tilde_XtX_inv_N @ (res_W_tilde[:, np.newaxis] * tilde_X).T # (p, N)
        hat_W_tilde = grad_W_tilde @ grad_W_tilde.T / N

        est_var = (hat_W_tilde / N) + (hat_W / n)

        return hat_theta, hat_W / n, est_var, hat_beta 

    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")


def LR_coef_target(theta, bootstrap, X, y, n, N=None, y_hat=None, weight=None, ppi=None, **kwargs):
    d = X.shape[1]
    length = X.shape[0]
    record = np.zeros((bootstrap, d))

    if ppi is None:
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n, replace=True)
            X_lbl = X[indices]
            y_lbl = y[indices]

            # Compute (X^T X)^{-1} projection to match sandwich variance estimator in inference
            XtX_inv_n = np.linalg.inv(X_lbl.T @ X_lbl / n)

            # Residuals
            residuals = (np.dot(X_lbl, theta) - y_lbl)

            # Scores: x_i * e_i -> (n, p)
            scores = residuals[:, np.newaxis] * X_lbl

            # Compute sandwich gradient: (X^T X)^{-1} @ (1/n sum scores)
            record[i] = XtX_inv_n @ (scores.sum(axis=0) / n)
        return np.cov(record, rowvar=False)
    
    if ppi == "single":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            X_lbl = X[indices[:n]]
            y_lbl = y[indices[:n]]
            y_hat_lbl = y_hat[indices[:n]]

            tilde_X = X[indices[n:]]
            tilde_y_hat = y_hat[indices[n:]]

            # Compute (X^T X)^{-1} projection to match variance calculation in inference
            XtX_inv_n = np.linalg.inv(X_lbl.T @ X_lbl / n)
            tilde_XtX_inv_N = np.linalg.inv(tilde_X.T @ tilde_X / N)

            # Unlabeled gradient: (X^T X)^{-1} @ X^T @ residuals
            res_unlabeled = (np.dot(tilde_X, theta) - tilde_y_hat)
            grad_unlabeled = tilde_XtX_inv_N @ (tilde_X.T @ res_unlabeled / N)

            # Labeled gradient: (X^T X)^{-1} @ X^T @ residuals
            res_labeled = (y_lbl - y_hat_lbl)
            grad_labeled = XtX_inv_n @ (X_lbl.T @ res_labeled / n)

            record[i] = grad_unlabeled - grad_labeled
        return np.cov(record, rowvar=False)
    elif ppi == "MoEs":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            X_lbl = X[indices[:n]]
            y_lbl = y[indices[:n]]
            y_hat_lbl = weight @ y_hat[:, indices[:n]]

            tilde_X = X[indices[n:]]
            tilde_y_hat = weight @ y_hat[:, indices[n:]]

            # Compute (X^T X)^{-1} projection to match variance calculation in inference
            XtX_inv_n = np.linalg.inv(X_lbl.T @ X_lbl / n)
            tilde_XtX_inv_N = np.linalg.inv(tilde_X.T @ tilde_X / N)

            # Unlabeled gradient: (X^T X)^{-1} @ X^T @ residuals
            res_unlabeled = (np.dot(tilde_X, theta) - tilde_y_hat)
            grad_unlabeled = tilde_XtX_inv_N @ (tilde_X.T @ res_unlabeled / N)

            # Labeled gradient: (X^T X)^{-1} @ X^T @ residuals
            res_labeled = (y_lbl - y_hat_lbl)
            grad_labeled = XtX_inv_n @ (X_lbl.T @ res_labeled / n)

            record[i] = grad_unlabeled - grad_labeled
        return np.cov(record, rowvar=False)
    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")


def Logistic_coef_inference(
    X: np.ndarray, 
    y: np.ndarray, 
    y_hat: Optional[np.ndarray] = None, 
    tilde_X: Optional[np.ndarray] = None, 
    tilde_y_hat: Optional[np.ndarray] = None, 
    ppi: Optional[str] = None,
    max_iter: int = 1000,
    epsilon: float = 1e-6,
    theta_init: Optional[np.ndarray] = None,
    **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimates the Logistic Regression Coefficient with optional Prediction-Powered Inference (PPI) corrections.
    """
    
    n = X.shape[0]
    p = X.shape[1]
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    
    
    if ppi is None:
        def zero_gradient_conventional(theta, X, y):
            z = np.dot(X, theta)
            p = sigmoid(z)
            return np.linalg.norm(((p-y) @ X))

        result = minimize(
            fun=zero_gradient_conventional,
            x0=np.zeros(p),
            args=(X, y),
            method='BFGS'
        )

        theta = result.x
        sigma_pred = sigmoid(X @ theta)
        # Hessian of estimating equation: H = (1/n) X^T diag(σ') X
        sigma_prime = sigma_pred * (1 - sigma_pred)
        H = (X.T * sigma_prime) @ X / n
        H_inv = np.linalg.pinv(H)
        # Sandwich variance: H^{-1} Σ_g H^{-1} / n
        residual = (sigma_pred - y)[:, np.newaxis] * X
        Sigma_g = residual.T @ residual / n
        return theta, H_inv @ Sigma_g @ H_inv / n

    elif ppi == "single":
        N = tilde_X.shape[0]
        def zero_gradient_single(theta, X, y, y_hat, tilde_X, tilde_y_hat):
            '''
            theta: the coefficient of the conventional model, (p, )
            X: (n, p), y: (n, ), y_hat: (n, ), tilde_X: (N, p), tilde_y_hat: (N, )
            '''
            tilde_z = np.dot(tilde_X, theta)
            tilde_p = sigmoid(tilde_z)
            return np.linalg.norm((tilde_p-tilde_y_hat) @ tilde_X / N - (y - y_hat) @ X / n)

        result = minimize(
            fun=zero_gradient_single,
            x0=np.zeros(p),
            args=(X, y, y_hat, tilde_X, tilde_y_hat),
            method='BFGS'
        )

        theta = result.x
        # Hessian from unlabeled data (derivative of PPI estimating eq w.r.t. θ)
        tilde_sigma = sigmoid(tilde_X @ theta)
        tilde_sigma_prime = tilde_sigma * (1 - tilde_sigma)
        H = (tilde_X.T * tilde_sigma_prime) @ tilde_X / N
        H_inv = np.linalg.pinv(H)

        # Labeled variance component (rectifier)
        residual_delta = (y - y_hat)[:, np.newaxis] * X
        Sigma_delta = residual_delta.T @ residual_delta / n

        # Unlabeled variance component
        residual_g = (tilde_sigma - tilde_y_hat)[:, np.newaxis] * tilde_X
        Sigma_g = residual_g.T @ residual_g / N

        # Sandwich correction
        Var_delta = H_inv @ Sigma_delta @ H_inv / n
        Var_total = H_inv @ (Sigma_g / N + Sigma_delta / n) @ H_inv
        return theta, Var_delta, Var_total

    elif ppi == "MoEs":
        N = tilde_X.shape[0]
        K = y_hat.shape[0]
        # --- 1. 预计算阶段 (Pre-computation) ---
        # 避免在循环中或定义中进行 O(N^2) 或 O(n^2) 的计算
        
        # 技巧：计算 (X @ X.T) 的对角线元素，复杂度 O(np) 而不是 O(n^2p)
        # diag_XX[i] = sum(X[i, :]^2)
        diag_XX = np.sum(X**2, axis=1)          # (n, )
        diag_tilde_XX = np.sum(tilde_X**2, axis=1) # (N, )

        # 预先计算小的中间矩阵 (K, p)
        y_hat_X = y_hat @ X
        tilde_y_hat_tilde_X = tilde_y_hat @ tilde_X
        
        # 预先计算 X.T @ y (p, )
        XT_y = X.T @ y

        def matrix_square_term(A):
            # A @ A.T 的优化替代
            return A @ A.T

        # 计算 hat_H (K, K)
        # 原式: y_hat @ diag(XXT) @ y_hat.T / n
        # 优化: (y_hat * diag_XX) @ y_hat.T / n  <-- 利用广播机制避免生成大对角阵
        term1_H = (y_hat * diag_XX) @ y_hat.T / n
        term2_H = matrix_square_term(y_hat_X / n)
        hat_H = term1_H - term2_H

        # 计算 hat_tilde_H (K, K)
        term1_tH = (tilde_y_hat * diag_tilde_XX) @ tilde_y_hat.T / N
        term2_tH = matrix_square_term(tilde_y_hat_tilde_X / N)
        hat_tilde_H = term1_tH - term2_tH

        # 计算 hat_r (K, )
        # 原式: y_hat @ diag @ y / n - y_hat @ X @ X.T @ y / n^2
        # 优化: 结合律调整 -> y_hat @ (X @ (X.T @ y))
        term1_r = (y_hat * diag_XX) @ y / n
        term2_r = y_hat_X @ XT_y / (n**2)
        hat_r = term1_r - term2_r

        # --- 2. 定义优化所需的函数 ---

        def residuals(theta, tilde_X, tilde_y_hat_agg, grad_labeled, N):
            """残差函数"""
            tilde_z = tilde_X @ theta
            tilde_p = expit(tilde_z) # 使用 scipy 的 sigmoid
            
            # (N, ) * (N, p) -> (p, )，利用广播避免构建对角阵
            # grad_unlabeled = tilde_X.T @ (tilde_p - tilde_y_hat_agg) / N
            # 上面这行等价于下面这行，但通常 einsum 或者 dot 更直观
            grad_unlabeled = (tilde_p - tilde_y_hat_agg) @ tilde_X / N
            
            return grad_unlabeled - grad_labeled

        def jacobian(theta, tilde_X, tilde_y_hat_agg, grad_labeled, N):
            """解析 Jacobian 矩阵 (p, p)，加速 least_squares"""
            tilde_z = tilde_X @ theta
            sig = expit(tilde_z)
            weights = sig * (1 - sig) # Sigmoid 导数: p(1-p)
            
            # J = (1/N) * X.T @ diag(weights) @ X
            # 优化计算: (X.T * weights) @ X
            J = (tilde_X.T * weights) @ tilde_X / N
            return J

        # --- 3. 迭代循环 ---
        
        prev_weight = np.ones(K) / K
        weight = prev_weight.copy()
        theta = theta_init if theta_init is not None else np.zeros(p) # 保持 theta 状态

        # 预计算线性方程组左边的矩阵 (K, K)，因为它不随迭代变化
        # Linear System: A * weight = b
        A_matrix = hat_H + (n/N) * hat_tilde_H
        
        total_progress = np.round(-np.log10(epsilon), 2)
        
        with tqdm(total=total_progress, desc="Optimization (Accelerated)", leave=False) as pbar:
            for _ in range(max_iter):
                # A. 更新 Theta
                y_hat_agg = weight @ y_hat         # (n, )
                tilde_y_hat_agg = weight @ tilde_y_hat # (N, )
                
                # 常量梯度项 (p, )
                grad_labeled = (y - y_hat_agg) @ X / n
                
                # 使用 Warm Start (x0=theta) 和 解析 Jacobian
                res = least_squares(
                    fun=residuals,
                    x0=theta, # <--- 关键优化：热启动
                    jac=jacobian, # <--- 关键优化：解析梯度
                    args=(tilde_X, tilde_y_hat_agg, grad_labeled, N),
                    method='lm' # Levenberg-Marquardt 通常最快
                    # method='trf' # 如果 lm 内存不够，换 trf
                )
                theta = res.x

                # B. 更新 Weight
                tilde_mu_theta = expit(tilde_X @ theta) # (N, )
                
                # 计算 hat_tilde_r (K, )
                # 原逻辑: pre_hat_tilde_r @ tilde_mu_theta
                # 原 pre_hat_tilde_r 包含巨大的 N*N 矩阵，这里直接用分解形式计算
                
                # Term 1: tilde_y_hat @ diag(tilde_XX) @ tilde_mu_theta / N
                # 优化: (tilde_y_hat * diag_tilde_XX) @ tilde_mu_theta / N
                r_term1 = (tilde_y_hat * diag_tilde_XX) @ tilde_mu_theta / N
                
                # Term 2: tilde_y_hat @ tilde_X @ tilde_X.T @ tilde_mu_theta / N^2
                # 优化: tilde_y_hat_tilde_X @ (tilde_X.T @ tilde_mu_theta) / N^2
                # 顺序: (K, p) @ (p, ) -> (K, )
                r_term2 = tilde_y_hat_tilde_X @ (tilde_X.T @ tilde_mu_theta) / (N**2)
                
                hat_tilde_r = r_term1 - r_term2
                
                # 解线性方程
                rhs = hat_r + (n/N) * hat_tilde_r
                new_weight = np.linalg.solve(A_matrix, rhs).flatten()

                # C. 检查收敛
                dist = np.linalg.norm(new_weight - prev_weight)
                
                # 更新进度条 (防止 log(0))
                if dist > 0:
                    current_prog = np.round(-np.log10(dist), 2)
                    if current_prog > pbar.n:
                        pbar.update(current_prog - pbar.n)
                
                weight = new_weight
                prev_weight = weight

                if dist <= epsilon:
                    break

        # --- 4. 后处理 ---
        y_hat_final = weight @ y_hat
        tilde_y_hat_final = weight @ tilde_y_hat

        # Hessian from unlabeled data
        mu_theta = expit(tilde_X @ theta)
        tilde_sigma_prime = mu_theta * (1 - mu_theta)
        H = (tilde_X.T * tilde_sigma_prime) @ tilde_X / N
        H_inv = np.linalg.pinv(H)

        # Labeled variance component (rectifier)
        resid_delta = (y - y_hat_final)[:, np.newaxis] * X
        Sigma_delta = resid_delta.T @ resid_delta / n

        # Unlabeled variance component
        resid_g = (mu_theta - tilde_y_hat_final)[:, np.newaxis] * tilde_X
        Sigma_g = resid_g.T @ resid_g / N

        # Sandwich correction
        Var_delta = H_inv @ Sigma_delta @ H_inv / n
        Var_total = H_inv @ (Sigma_g / N + Sigma_delta / n) @ H_inv
        return theta, Var_delta, Var_total, weight


def Logistic_coef_target(theta, bootstrap, X, y, n, N=None, y_hat=None, weight=None, ppi=None):
    d = X.shape[1]
    length = X.shape[0]
    record = np.zeros((bootstrap, d))


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    if ppi is None:
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n, replace=True)
            X_lbl = X[indices]
            y_lbl = y[indices]
            record[i] = (sigmoid(np.dot(X_lbl, theta)) - y_lbl) @ X_lbl / n
        return np.cov(record, rowvar=False)
    
    if ppi == "single":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            X_lbl = X[indices[:n]]
            y_lbl = y[indices[:n]]
            y_hat_lbl = y_hat[indices[:n]]

            tilde_X = X[indices[n:]]
            tilde_y_hat = y_hat[indices[n:]]    

            # print(y_lbl.shape, y_hat_lbl.shape, X_lbl.shape, tilde_X.shape, tilde_y_hat.shape)
            record[i] = (sigmoid(np.dot(tilde_X, theta)) - tilde_y_hat) @ tilde_X / N - (y_lbl - y_hat_lbl) @ X_lbl / n
        return np.cov(record, rowvar=False)
    elif ppi == "MoEs":
        for i in range(bootstrap):
            indices = np.random.choice(length, size=n+N, replace=True)
            X_lbl = X[indices[:n]]
            y_lbl = y[indices[:n]]
            y_hat_lbl = weight @ y_hat[:, indices[:n]]

            tilde_X = X[indices[n:]]
            tilde_y_hat = weight @ y_hat[:, indices[n:]]

            record[i] = (sigmoid(np.dot(tilde_X, theta)) - tilde_y_hat) @ tilde_X / N - (y_lbl - y_hat_lbl) @ X_lbl / n
        return np.cov(record, rowvar=False)
    else:
        raise ValueError(f"Unknown ppi mode: {ppi}")