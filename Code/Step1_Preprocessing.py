# Data Loading and Robust Feature Selection for Gene Network Construction
# This script loads and preprocesses gene expression data, standardizes age information, and selects informative genes based on variance and mutual information.


import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

SEED = 42
np.random.seed(SEED)

def load_data(filepath, var_percentile=95, mi_percentile=90):
    """Robust feature selection pipeline for gene expression data"""
    df = pd.read_csv(filepath)

    # Normalize age to months
    def convert_age(age_str):
        age_str = age_str.lower().strip()
        if 'mos' in age_str:
            return int(age_str.replace('mos', ''))
        elif 'yr' in age_str:
            return int(float(age_str.replace('yrs', '').replace('yr', '')) * 12)
        else:
            raise ValueError(f"Invalid age format: {age_str}")
    
    df['age_months'] = df['Age'].apply(convert_age)
    features = df.drop(columns=['Age', 'Sample', 'age_months'])
    target = df['age_months']
    
    # Variance filtering
    variances = features.var(axis=0)
    var_selector = VarianceThreshold(threshold=np.nanpercentile(variances, var_percentile))
    var_selector.fit(features)
    var_selected = features.columns[var_selector.get_support()]

    # Mutual information filtering
    mi_scores = mutual_info_regression(features[var_selected], target, random_state=SEED)
    final_genes = var_selected[mi_scores >= np.nanpercentile(mi_scores, mi_percentile)]
    
    print(f"Number of retained genes: {len(final_genes)}")
    return features[final_genes], final_genes.tolist()
