class Inference:
    """ Base class for inference methods. """

    def __init__(self):
        pass

    def requires_class_label(self):
        raise NotImplementedError()

    def requires_joint_ratio(self):
        raise NotImplementedError()

    def requires_joint_score(self):
        raise NotImplementedError()

    def predicts_density(self):
        return NotImplementedError()

    def predicts_ratio(self):
        return NotImplementedError()

    def predicts_score(self):
        return NotImplementedError()

    def fit(self, theta=None, x=None, y=None, r_xz=None, t_xz=None,
            batch_size=64, initial_learning_rate=0.001, final_learning_rate=0.0001, n_epochs=50):
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
