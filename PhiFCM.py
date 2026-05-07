import torch


class PhiFCM:

    def __init__(self, k, alpha=1.0, m=2.0, max_iter=100, tol=1e-4, device=None):
        self.k = k
        self.alpha = alpha
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Phi transform
    @staticmethod
    def phi(x, alpha):
        return torch.sign(x) * torch.abs(x) ** alpha

    # Initialize memberships randomly
    def init_membership(self, n):
        U = torch.rand(n, self.k, device=self.device)
        U = U / U.sum(dim=1, keepdim=True)
        return U

    # Init from centroids (NEW)
    @staticmethod
    def init_from_centroids(Y, W):
        dist = torch.cdist(Y, W)
        labels = torch.argmin(dist, dim=1)
        U = torch.zeros(Y.shape[0], W.shape[0], device=Y.device)
        U[torch.arange(Y.shape[0]), labels] = 1.0
        return U

    # Update prototypes (fixed-point)
    def update_centroids(self, Y, U):
        q = 2.0 / self.alpha
        W = torch.zeros(self.k, Y.shape[1], device=self.device)
        for r in range(self.k):
            u = U[:, r] ** self.m
            w = torch.mean(Y, dim=0)
            for _ in range(10):
                diff = Y - w
                dist = torch.norm(diff, dim=1) + 1e-8
                weights = u * dist ** (q - 2)
                w = (weights[:, None] * Y).sum(dim=0) / weights.sum()
            W[r] = w
        return W

    # Update memberships
    def update_membership(self, Y, W):
        q = 2.0 / self.alpha
        dist = torch.cdist(Y, W) + 1e-8
        power = q / (self.m - 1)
        ratio = dist[:, :, None] / dist[:, None, :]
        U = 1.0 / (ratio ** power).sum(dim=2)
        return U

    # FS index
    def fukuyama_sugeno(self, Y, U, W):
        q = 2.0 / self.alpha
        um = U ** self.m
        dist = torch.cdist(Y, W) ** q
        term1 = (um * dist).sum()
        y_bar = Y.mean(dim=0)
        center_dist = torch.norm(W - y_bar, dim=1) ** q
        s_r = um.sum(dim=0)
        term2 = (s_r * center_dist).sum()
        return (term1 - term2).item()

    # Fit
    def fit(self, X, init_centroids=None):
        X = X.to(self.device)
        Y = self.phi(X, self.alpha)
        n = X.shape[0]

        # FIX: same initialization as KMeans if provided
        if init_centroids is not None:
            W = self.phi(init_centroids.to(self.device), self.alpha)
            U = self.init_from_centroids(Y, W)
        else:
            U = self.init_membership(n)

        for _ in range(self.max_iter):
            W = self.update_centroids(Y, U)
            U_new = self.update_membership(Y, W)

            if torch.norm(U_new - U) < self.tol:
                break

            U = U_new

        self.U = U
        self.W = W
        self.labels_ = torch.argmax(U, dim=1)
        return self

    # Predict
    def predict(self, X):
        X = X.to(self.device)
        Y = self.phi(X, self.alpha)
        dist = torch.cdist(Y, self.W)
        return torch.argmin(dist, dim=1)