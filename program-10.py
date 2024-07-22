import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
def initialize_parameters(X, n_components):
    np.random.seed(0)
    n_samples, n_features = X.shape
    means = X[np.random.choice(n_samples, n_components, False)]
    covariances = [np.cov(X, rowvar=False)] * n_components
    weights = np.ones(n_components) / n_components
    return means, covariances, weights
def e_step(X, means, covariances, weights):
    n_samples, n_features = X.shape
    n_components = len(weights)
    responsibilities = np.zeros((n_samples, n_components))
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities
def m_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    means = np.zeros((n_components, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    weights = np.zeros(n_components)
    for k in range(n_components):
        N_k = responsibilities[:, k].sum()
        means[k] = (X * responsibilities[:, k][:, np.newaxis]).sum(axis=0) / N_k
        covariances[k] = ((X - means[k]).T @ np.diag(responsibilities[:, k]) @ (X - means[k])) / N_k
        weights[k] = N_k / n_samples
    return means, covariances, weights
def em_algorithm(X, n_components, n_iter=100):
    means, covariances, weights = initialize_parameters(X, n_components)
    for _ in range(n_iter):
        responsibilities = e_step(X, means, covariances, weights)
        means, covariances, weights = m_step(X, responsibilities)
    return means, covariances, weights
if __name__ == "__main__":
    np.random.seed(0)
    X1 = np.random.randn(300, 2) + np.array([5, 5])
    X2 = np.random.randn(300, 2) + np.array([-5, -5])
    X = np.vstack([X1, X2])
    n_components = 2
    means, covariances, weights = em_algorithm(X, n_components) 
    plt.scatter(X[:, 0], X[:, 1], c='grey', label='Data points')
    for k in range(n_components):
        plt.scatter(means[k][0], means[k][1], marker='x', s=100, label=f'Cluster {k + 1}')
    plt.title('EM Algorithm for Gaussian Mixture Model')
    plt.legend()
    plt.show()
