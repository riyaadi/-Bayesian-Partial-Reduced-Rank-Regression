Bayesian Partial Reduced-Rank Regression (BPRR)
This package implements Bayesian Partial Reduced-Rank Regression (BPRR) with automatic grouping of responses, rank selection, and full posterior inference via MCMC. Below, every function's input and output are clearly described, along with their purpose in the workflow.


 — How to Run
1. Requirements
·        Python 3.x
·        Required packages: numpy, scipy, pandas, scikit-learn, matplotlib, seaborn
·        (Optional) Jupyter Notebook or Google Colab for interactive use
         .     Run all the functions code block given in uploaded python notebook 


—for applying model to pre covid data and post covid data uncomment wherever written in code block given in python notebook(clearly texted in python notebook to specify that code block)
2. Prepare Your Data
·        Format:
o   X: Design matrix (shape: n_samples × n_predictors)
o   Y: Response matrix (shape: n_samples × n_responses)
·        File type:
o   Accepts .xlsx (Excel), .csv, or can be generated/simulated in code.
·        Example:
o   For real data, edit the following lines to load your data:
import pandas as pd
macro = pd.read_excel("macroUS.xlsx")
Y = macro.iloc[:, 0:5].values
X = macro.iloc[:, 5:10].values


o   For simulated data, see the provided simulation block in the notebook.
3. Preprocess the Data
·        Standardize both X and Y:
from sklearn.preprocessing import StandardScaler
scaler_Y = StandardScaler()
scaler_X = StandardScaler()
Y = scaler_Y.fit_transform(Y)
X = scaler_X.fit_transform(X)


4. Set Initial Model Parameters
·        Choose an initial gamma vector (binary, length = number of responses):
import numpy as np
gamma_init = np.array([1, 1, 0, 0, 0])  # Example: first two responses are low-rank


5. Run the BPRR Sampler
·        Call the main sampler function:
results = sampler_bprr(
        Y=Y,
        X=X,
        mcmc=600,         # Number of MCMC samples to keep
        burnin=300,   # Number of burn-in iterations
        thin=1,           # Thinning interval
        iprint=10,        # Print progress every 10 iterations
        gamma=gamma_init
)


·        Adjust mcmc, burnin, and thin as needed for your data size and desired accuracy.
6. Postprocess the Results
·        Summarize posterior draws and compute estimates:
OUT2 = postprocess(Y, X, results)


·        This returns posterior means and summary statistics.
7. Visualize Results
·        Posterior probabilities of gamma patterns:
# See notebook for plotting code


·        Estimated coefficient matrix heatmap:
# See notebook for plotting code


·        Posterior distribution of rank r:
# See notebook for plotting code


8. Tips for Custom Data
·        Make sure the number of columns in X and Y match the model's expectations.
·        If using a different data file, update the path and column indices accordingly.
·        All functions are modular; you can adapt the data loading and preprocessing sections as needed.
9. Troubleshooting
·        If you encounter errors about matrix dimensions, double-check your input shapes.
·        For singular matrix errors, ensure your predictors are not perfectly collinear.
·        For large data sets, consider increasing thin to reduce memory usage.
Summary:
1.          Install dependencies
2.         Prepare and standardize your data (X, Y)
3.          Set initial gamma
4.         Run sampler_bprr
5.          Postprocess and visualize results
For more details, see the code comments and example blocks in the notebook


Function Descriptions
gamma_candidates
Purpose:
Generate all valid candidate binary vectors ("gamma") for possible groupings of responses.
Input:
* gamma (1D numpy array): Current binary vector (length q) indicating group assignment.
Output:
* gammas (2D numpy array): Each row is a valid candidate gamma vector.
logsumexp
Purpose:
Compute the log-sum-exp and softmax of a vector in a numerically stable way.
Input:
* x (1D numpy array): Input vector.
Output:
* lse (float): Log-sum-exp of input vector.
* sm (1D numpy array): Softmax of input vector.
logdet
Purpose:
Compute the logarithm of the determinant of a square matrix.
Input:
* A (2D numpy array): Square matrix.
* method (str or None): If 'chol', uses Cholesky decomposition (for positive-definite matrices); otherwise uses LU.
Output:
* (float): Log-determinant of the matrix. Returns  −∞ for singular matrices.
sample_phi_slice
Purpose:
Sample the AR(1) parameter phi using slice sampling on a discrete grid.
Inputs:
* phih (float): Current value of phi.
* h (1D numpy array): Time series vector.
* muh (float): Mean parameter.
* sig2h (float): Variance parameter.
* a0, b0 (float): Beta prior parameters.
Outputs:
* phih (float): New sampled phi.
* uu (float): Auxiliary variable used for slice sampling.
HansenIMLE
Purpose:
Estimate the low-rank coefficient matrix using Hansen's Iterative Maximum Likelihood Estimation (IMLE) algorithm.
Inputs:
* Y (n × q array): Response matrix.
* X (n × p array): Predictor matrix.
* r (int): Rank parameter.
* qg (int): Number of low-rank responses.
* SigmaHinv (matrix): Precomputed inverse covariance matrix.
* tol (float, optional): Convergence tolerance (default: 1e-4).
* maxIter (int, optional): Maximum iterations (default: 500).
Output:
* C1hat (p × qg array): Estimated low-rank coefficient matrix.
posterior_gamma
Purpose:
Compute posterior probabilities for all candidate gamma vectors, marginalizing over possible ranks.
Inputs:
* Y (n × q array): Response matrix.
* X (n × p array): Predictor matrix.
* Sigma (q × q array): Covariance matrix.
* d (float): Scalar for prior variance.
* gamma (1D array): Current gamma vector.
* rho (float): Prior parameter for gamma.
* Knq, Kqn (arrays): Commutation matrices for vectorization.
Outputs:
* ptilde (1D array): Posterior probabilities for each candidate gamma.
* lognum (1D array): Log numerators for each candidate.
* gammas (2D array): Candidate gamma matrix.
* logf_r (2D array): Laplace approximations for each gamma and rank.
reorder
Purpose:
Reorder responses, coefficient matrices, and covariance matrices according to the current gamma vector.
Inputs:
* Y (n × q array): Response matrix.
* X (n × p array): Predictor matrix.
* C (p × q array): Coefficient matrix.
* Sigma (q × q array): Covariance matrix.
* d (float): Scalar for prior variance.
* gamma (1D array): Current gamma vector.
Outputs:
* Y (n × q array): Reordered response matrix.
* y (nq × 1 array): Vectorized reordered responses.
* C (p × q array): Reordered coefficient matrix.
* Sigma (q × q array): Reordered covariance matrix.
* Sigmatilde (nq × nq array): Kronecker covariance.
* U1, U2 (matrices): Design matrices for low- and full-rank groups.
* Sigma_delta (matrix): Prior variance for full-rank group.
* qg (int): Number of low-rank responses.
* q2 (int): Number of full-rank responses.
* ylabel (1D array): Indices of reordered responses.
sampler_bprr
Purpose:
Run the full MCMC sampler for Bayesian Partial Reduced-Rank Regression.
Inputs:
* Y (n × q array): Response matrix.
* X (n × p array): Predictor matrix.
* mcmc (int): Number of MCMC samples to keep.
* burnin (int): Number of burn-in iterations.
* thin (int): Thinning interval.
* iprint (int): Print progress every iprint iterations.
* gamma (optional, 1D array): Initial gamma vector.
Output:
* OUT (dict): Contains arrays of sampled gamma, rank, coefficients, covariance, predictive draws, and more:
   * 'store_gamma': (N, q) array of sampled gamma vectors
   * 'store_r': (N,) array of sampled ranks
   * 'store_A', 'store_B', 'store_C2': Arrays of sampled coefficient matrices
   * 'store_Sigma': (N, q, q) array of sampled covariance matrices
   * 'store_Ypred': (N, n, q) array of predictive draws
   * 'store_acc': (N,) array of acceptance rates
   * 'store_ylabel': (N, q) array of response orderings
postprocess
Purpose:
Post-process MCMC output to compute posterior summaries: modal gamma, modal rank, posterior mean coefficients, covariance, and prediction error.
Inputs:
* Y (n × q array): Response matrix.
* X (n × p array): Predictor matrix.
* OUT (dict): MCMC output from sampler_bprr.
* burnin (int, optional): Number of initial samples to discard (default: 0).
Output:
* results (dict):
   * 'qg': Number of low-rank responses
   * 'gamma': Modal gamma vector
   * 'rhat': Modal rank
   * 'Chat': Posterior mean coefficient matrix (p × q)
   * 'Chat2': Reordered coefficient matrix (p × q)
   * 'Sigmahat': Posterior mean covariance (q × q)
   * 'Sigmahat2': Reordered covariance (q × q)
   * 'idx_keep', 'idx_keep2': Indices of kept samples
   * 'mse': Mean squared error of predictions
All functions are modular and can be tested independently. For more information, see the docstrings in each function.