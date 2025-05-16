import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.datasets import load_iris



class Correlation:
  def __init__(self):
    print("Constructor")

  def pearson_corr(self,X):
    nexamples,nfeatures = X.shape
    n = nexamples
    self.corr_matrix = np.zeros((nfeatures,nfeatures))
    mean = np.mean(X,axis = 0)
    for i in range(nfeatures):
      y = np.copy(X)
      x = X[:,i]
      x = x.reshape(-1,1)
      sum_x = np.sum(x,axis=0)
      sum_y = np.sum(y,axis = 0)
      sum_xy = np.sum(x * y,axis = 0)
      sum_x_sum_y = sum_x * sum_y
      x_square_sum = np.sum(x**2,axis = 0)
      sum_x_square = sum_x ** 2
      y_square_sum = np.sum(y**2,axis = 0)
      sum_y_square = sum_y ** 2
      numerator = (n * sum_xy - sum_x_sum_y)
      denominator =  np.sqrt(( n * x_square_sum - sum_x_square) * (n * y_square_sum - sum_y_square))
      r =  numerator / denominator
      self.corr_matrix[i,:] = r

    return self.corr_matrix

  def spearman_corr(self,X):
    n_samples, n_features = X.shape
    corr_matrix = np.zeros((n_features, n_features))

    # Convert data to ranks
    X_ranked = np.apply_along_axis(rankdata, axis=0, arr=X)

    for i in range(n_features):
        x = X_ranked[:, i]
        x = x.reshape(-1,1)
        y = np.copy(X_ranked)
        d = x - y
        d_squared_sum = np.sum(d**2,axis = 0)

        # Apply Spearman formula
        n = len(x)
        rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        corr_matrix[i,:] = rho

    return corr_matrix







# c = Correlation()
# pearson_matrix = c.pearson_corr(X)
# spearman_matrix = c.spearman_corr(X)
# spearman_matrix


