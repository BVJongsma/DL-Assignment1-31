import sklearn.datasets

def load_data():
    N = 500
    gq = sklearn.datasets.make_gaussian_quantiles(
            mean=None,
            cov=0.7,
            n_samples=N,
            n_features=2,
            n_classes=2,
            shuffle=True,
            random_state=None)
    return gq

learning_rate = 0.1