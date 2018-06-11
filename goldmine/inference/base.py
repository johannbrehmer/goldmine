class Inference:

    """ Base class for inference methods. """

    def __init__(self):
        raise NotImplementedError()

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def predict_density(self, x=None, theta=None):
        raise NotImplementedError()

    def predict_ratio(self, x=None, theta=None, theta1=None):
        raise NotImplementedError()

    def predict_score(self, x=None, theta=None):
        raise NotImplementedError()
