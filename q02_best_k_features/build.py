# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    y = data.pop('SalePrice')
    X = data
    sel_feat = SelectPercentile(score_func=f_regression, percentile=k)
    sel_feat.fit_transform(X, y)
    n = sel_feat.get_support(indices=True).size
    return list(X.columns[np.argsort(sel_feat.scores_)[::-1]][:n])
