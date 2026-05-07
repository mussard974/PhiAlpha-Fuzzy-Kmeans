import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score


class PhiKMeans:

    def __init__(self, k, alpha=1.0, max_iter=100, centroid_iter=10, tol=1e-4, device=None):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.centroid_iter = centroid_iter
        self.tol = tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Phi transform
    @staticmethod
    def phi(x, alpha):
        return torch.sign(x) * torch.abs(x) ** alpha

    # Pairwise d_phi distance
    @staticmethod
    def d_phi_pairwise(X, Y, alpha):
        X_phi = PhiKMeans.phi(X, alpha)
        Y_phi = PhiKMeans.phi(Y, alpha)
        diff = X_phi[:, None, :] - Y_phi[None, :, :]
        dist = torch.norm(diff, dim=-1)
        return dist ** (1.0 / alpha)

    # KMeans++ initialization
    @staticmethod
    def kmeans_pp_init(X, k, seed=0):
        torch.manual_seed(seed)
        n = X.shape[0]
        centroids = []
        idx = torch.randint(0, n, (1,))
        centroids.append(X[idx])
        for _ in range(1, k):
            dist = torch.cdist(X, torch.cat(centroids))
            min_dist, _ = torch.min(dist, dim=1)
            probs = min_dist**2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            centroids.append(X[idx])
        return torch.cat(centroids)

    # Centroid update
    @staticmethod
    def update_centroid(Z, m, alpha, n_iter=10, eps=1e-8):
        for _ in range(n_iter):
            diff = Z - m
            dist = torch.norm(diff, dim=1) + eps
            w = dist ** (2.0 / alpha - 2.0)
            m = (w[:, None] * Z).sum(dim=0) / w.sum()
        return m

    # Fit
    def fit(self, X, init_centroids):
        X = X.to(self.device)
        Z = self.phi(X, self.alpha)
        centroids = self.phi(init_centroids.to(self.device), self.alpha)
        for _ in range(self.max_iter):
            dist = torch.cdist(Z, centroids)
            labels = torch.argmin(dist, dim=1)
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.k):
                mask = labels == k
                if mask.any():
                    Zk = Z[mask]
                    mk = centroids[k]
                    new_centroids[k] = self.update_centroid(
                        Zk, mk, self.alpha, self.centroid_iter
                    )
                else:
                    new_centroids[k] = Z[torch.randint(0, Z.shape[0], (1,))]
            shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break
        self.centroids_phi = centroids
        self.centroids = torch.sign(centroids) * torch.abs(centroids) ** (1 / self.alpha)
        self.labels_ = labels
        return self

    # Predict
    def predict(self, X):
        X = X.to(self.device)
        Z = self.phi(X, self.alpha)
        dist = torch.cdist(Z, self.centroids_phi)
        return torch.argmin(dist, dim=1)

    # Silhouette score
    @staticmethod
    def silhouette_score(X, labels, alpha):
        D = PhiKMeans.d_phi_pairwise(X, X, alpha)
        n = X.shape[0]
        sil = torch.zeros(n, device=X.device)
        unique_labels = labels.unique()
        for i in range(n):
            same = labels == labels[i]
            if same.sum() > 1:
                a = D[i, same].sum() / (same.sum() - 1)
            else:
                a = torch.tensor(0.0, device=X.device)
            b_vals = []
            for l in unique_labels:
                if l != labels[i]:
                    mask = labels == l
                    b_vals.append(D[i, mask].mean())
            b = torch.min(torch.stack(b_vals))
            sil[i] = (b - a) / torch.maximum(a, b)
        return sil.mean().item()

    # Hungarian + metrics
    @staticmethod
    def clustering_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        cost = np.zeros((D, D))
        for i in range(len(y_true)):
            cost[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(-cost)
        mapping = {r: c for r, c in zip(row_ind, col_ind)}
        y_aligned = np.array([mapping[l] for l in y_pred])
        return {
            "precision": precision_score(y_true, y_aligned, average="macro"),
            "recall": recall_score(y_true, y_aligned, average="macro"),
            "f1": f1_score(y_true, y_aligned, average="macro"),
        }

    # Alpha grid search
    @staticmethod
    def alpha_search(X, k, alphas, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        init_centroids = PhiKMeans.kmeans_pp_init(X, k)
        best_alpha = None
        best_score = -1
        best_model = None
        for alpha in alphas:
            model = PhiKMeans(k, alpha, device=device)
            model.fit(X, init_centroids)
            score = PhiKMeans.silhouette_score(X, model.labels_, alpha)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_model = model
        return best_alpha, best_model