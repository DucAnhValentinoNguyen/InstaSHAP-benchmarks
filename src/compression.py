import numpy as np
from sklearn.cluster import KMeans


def uniform_subsample(X, k, random_state=0):
    """
    Simple uniform subsampling of k rows from X.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=k, replace=False)
    return X.iloc[idx].reset_index(drop=True)


def kmeans_compression(X, k, random_state=0):
    """
    Compress dataset X into k cluster centers using k-means.

    This approximates the background distribution with k points.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X)

    centers = km.cluster_centers_
    return centers  # shape (k, d)


def gaussian_mixture_compression(X, k, random_state=0):
    """
    Optional alternative using GMM.
    """
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state)
    gm.fit(X)
    
    # Mean of each Gaussian component
    return gm.means_
