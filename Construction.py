# Stabilized Dynamic Gene Network Construction via ElasticNet and Bootstrapping
# This module builds a robust gene network by repeatedly applying ElasticNet regression on bootstrapped data and averaging the coefficients.

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm

SEED = 42

def build_dynamic_network(data, genes, n_bootstrap=50):
    """Bootstrapped ElasticNet regression for stable network construction"""
    X = data[genes].values
    n_genes = len(genes)
    adj_matrix = np.zeros((n_genes, n_genes))

    for i in range(n_bootstrap):
        idx = np.random.RandomState(SEED + i).choice(X.shape[0], X.shape[0], replace=True)
        X_scaled = StandardScaler().fit_transform(X[idx])

        def _fit_gene(j):
            model = ElasticNetCV(
                l1_ratio=[0.5, 0.7, 0.9],
                alphas=np.logspace(-4, 2, 50),
                cv=5,
                max_iter=5000,
                tol=1e-4,
                random_state=SEED,
                selection='random'
            )
            try:
                model.fit(np.delete(X_scaled, j, axis=1), X_scaled[:, j])
                return np.insert(model.coef_, j, 0.0)
            except:
                return np.zeros(n_genes)
        
        coefs = Parallel(n_jobs=-1, prefer="threads")(delayed(_fit_gene)(j) for j in tqdm(range(n_genes), desc=f"Bootstrap {i+1}"))
        adj_matrix += np.array(coefs)

    adj_mean = adj_matrix / n_bootstrap
    nonzero_abs = np.abs(adj_mean[adj_mean != 0])
    threshold = np.nanpercentile(nonzero_abs, 95) if nonzero_abs.size > 0 else 0
    adj_mean[np.abs(adj_mean) < threshold] = 0

    return pd.DataFrame(adj_mean, index=genes, columns=genes)
