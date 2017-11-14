from sklearn.gaussian_process.kernels import RBF

class svmgo:

    def __init__(self):
        self.ensemble = None
        self.gamma_opt = True
        self._gamma = 'auto'
        self.alpha = None
        self.kernel = 'rbf'
        self.random_state = None
        self.niter = 1000
        self.epsilon = 0.001
        self.gd_method = 'ssgd'
        self.support_vectors_ = None
        self._supp_idx = []
        self.dropout = 0
        self.regularize = False
        self.C = 1

        # Metrics
        self.gamma_progress = []
        self.alpha_progress = []
        self.loss_progress = []
        self.support_progress = []


    def fit(self, X, y):
