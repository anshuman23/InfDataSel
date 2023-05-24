from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import random
import numpy as np
from tqdm import tqdm
import sklearn.neural_network
import sklearn.linear_model
import sklearn.metrics
from scipy.linalg import cho_solve, cho_factor

import torch
from torch import nn
from torch import Tensor

from dataset import fetch_data
from eval import Evaluator
from utils import nearest_pd


class IFBaseClass(ABC):
    """ Abstract base class for influence function computation on logistic regression """

    @staticmethod
    def set_sample_weight(n: int, sample_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == n
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == n
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0.
            assert max(sample_weight) <= 2.

        return sample_weight

    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True) -> np.ndarray:
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)

    @abstractmethod
    def log_loss(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
                 l2_reg: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             l2_reg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def grad_pred(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def hess(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             check_pos_def: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pred(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the predictive probability and class label """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> None:
        raise NotImplementedError


class LogisticRegression(IFBaseClass):
    """
    Logistic regression: pred = sigmoid(weight^T @ x + bias)
    Currently only support binary classification
    Borrowed from https://github.com/kohpangwei/group-influence-release
    """

    def __init__(self, l2_reg: float, fit_intercept: bool = False):
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.model = sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=(1. / l2_reg),
            fit_intercept=fit_intercept,
            tol=1e-8,
            solver="lbfgs",
            max_iter=2048,
            multi_class="ovr",
            warm_start=False,
        )

    def log_loss(self, x, y, sample_weight=None, l2_reg=False, eps=1e-16):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        pred, _, = self.pred(x)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            log_loss += self.l2_reg * np.linalg.norm(self.weight, ord=2) / 2.

        return log_loss

    def grad(self, x, y, sample_weight=None, l2_reg=False):
        """
        Compute the gradients: grad_wo_reg = (pred - y) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        indiv_grad = x * (pred - y).reshape(-1, 1)
        reg_grad = self.l2_reg * self.weight
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        if self.fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)
            reg_grad = np.concatenate([reg_grad, np.zeros(1)], axis=0)

        total_grad = np.sum(weighted_indiv_grad, axis=0)

        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)
        indiv_grad = x * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

        return total_grad, weighted_indiv_grad

    def hess(self, x, sample_weight=None, check_pos_def=False):
        """
        Compute hessian matrix: hessian = pred * (1 - pred) @ x^T @ x + lambda
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        factor = pred * (1. - pred)
        indiv_hess = np.einsum("a,ai,aj->aij", factor, x, x)
        reg_hess = self.l2_reg * np.eye(x.shape[1])

        if self.fit_intercept:
            off_diag = np.einsum("a,ai->ai", factor, x)
            off_diag = off_diag[:, np.newaxis, :]

            top_row = np.concatenate([indiv_hess, np.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = np.concatenate([off_diag, factor.reshape(-1, 1, 1)], axis=2)
            indiv_hess = np.concatenate([top_row, bottom_row], axis=1)

            reg_hess = np.pad(reg_hess, [[0, 1], [0, 1]], constant_values=0.)

        hess_wo_reg = np.einsum("aij,a->ij", indiv_hess, sample_weight)
        total_hess_w_reg = hess_wo_reg + reg_hess

        if check_pos_def:
            self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def fit(self, x, y, sample_weight=None, verbose=False):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        self.model.fit(x, y, sample_weight=sample_weight)
        self.weight: np.ndarray = self.model.coef_.flatten()
        if self.fit_intercept:
            self.bias: np.ndarray = self.model.intercept_

        if verbose:
            pred, _ = self.pred(x)
            train_loss_wo_reg = self.log_loss(x, y, sample_weight)
            reg_loss = np.sum(np.power(self.weight, 2)) * self.l2_reg / 2.
            train_loss_w_reg = train_loss_wo_reg + reg_loss

            print("Train loss: %.5f + %.5f = %.5f" % (train_loss_wo_reg, reg_loss, train_loss_w_reg))

        return

    def pred(self, x):
        return self.model.predict_proba(x)[:, 1], self.model.predict(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, factor=1):
        super(MLPClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // factor, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            nn.ReLU(),
        )

        self.last_fc = nn.Linear(input_dim // factor, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(self.last_fc(x))
        return x.squeeze()

    def emb(self, x):
        x = self.layer(x)
        return x

    @property
    def num_parameters(self):
        return self.last_fc.weight.nelement()


class MLPClassifier2(nn.Module):
    def __init__(self, input_dim, factor=4):
        super(MLPClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // factor, bias=False)
        self.fc2 = nn.Linear(input_dim // factor, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()

    @property
    def num_parameters(self):
        return self.fc1.weight.nelement() + self.fc2.weight.nelement()

    @property
    def weight(self):
        return torch.cat([self.fc1.weight.flatten(), self.fc2.weight.flatten()], dim=0)


class NN(IFBaseClass):
    """
    Neural Networks
    Currently only support binary classification
    """

    def __init__(self, input_dim, l2_reg, n_iter=10000, lr=1e-3, device="cuda:0", seed: int = None):
        super(NN, self).__init__(l2_reg)

        self.l2_reg = l2_reg
        self.n_iter = n_iter
        self.lr = lr
        self.input_dim = input_dim
        self.pred_threshold = 0.5
        self.device = torch.device(device)

        if seed is not None:
            self.fix_seed(seed)

    @staticmethod
    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def fit(self, x, y, sample_weight=None, save_path: str = None):
        """ Currently using full batch training """

        self.model = MLPClassifier2(self.input_dim).to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=self.l2_reg, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.l2_reg)

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        y = torch.from_numpy(y).to(torch.float).to(self.device)
        sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.iter // 4, gamma=0.1)

        for i in range(self.n_iter):
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.log_loss_tensor(y, pred, sample_weight)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

        return

    def pred(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x)
        pred_label = (pred > self.pred_threshold).to(torch.float)

        return pred.cpu().numpy(), pred_label.cpu().numpy()

    def grad(self, x, y, sample_weight=None, l2_reg=False):

        N = x.shape[0]
        sample_weight = self.set_sample_weight(N, sample_weight)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        y = torch.from_numpy(y).to(torch.float).to(self.device)
        sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        weighted_indiv_grad = np.zeros((N, self.model.num_parameters))
        pred = self.model(x)
        for i, (single_pred, single_y, single_weight) in enumerate(
                tqdm(zip(pred, y, sample_weight), total=N, desc="Computing individual first-order gradient")):
            indiv_loss = self.log_loss_tensor(single_y, single_pred, single_weight)
            indiv_grad = torch.autograd.grad(indiv_loss, self.model.parameters(), retain_graph=True)
            indiv_grad = torch.cat([x.flatten() for x in indiv_grad], dim=0).detach().cpu().numpy()
            weighted_indiv_grad[i] = indiv_grad

        total_grad = np.sum(weighted_indiv_grad, axis=0)
        reg_grad = self.l2_reg * self.model.weight
        reg_grad = reg_grad.detach().cpu().numpy()

        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def hess(self, x, y, sample_weight=None, check_pos_def=True) -> np.ndarray:
        """
        Compute hessian matrix for the whole training set
        """

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        x = torch.from_numpy(x).to(torch.float).to(self.device)
        y = torch.from_numpy(y).to(torch.float).to(self.device)
        sample_weight = torch.from_numpy(sample_weight).float().to(self.device)

        pred = self.model(x)
        loss = self.log_loss_tensor(y, pred, sample_weight)
        hess_wo_reg = torch.zeros((self.model.num_parameters, self.model.num_parameters)).to(self.device)

        grad = torch.autograd.grad(outputs=loss, inputs=self.model.parameters(), create_graph=True)
        grad = torch.cat([x.flatten() for x in grad], dim=0)

        for i, g in enumerate(tqdm(grad, total=self.model.num_parameters, desc="Computing second-order gradient")):
            second_order_grad = torch.autograd.grad(outputs=g, inputs=self.model.parameters(), retain_graph=True)
            second_order_grad = torch.cat([x.flatten() for x in second_order_grad], dim=0)
            hess_wo_reg[i, :] = second_order_grad

        hess_wo_reg = hess_wo_reg.detach().cpu().numpy()
        reg_hess = self.l2_reg * np.eye(self.model.num_parameters)

        total_hess_w_reg = hess_wo_reg + reg_hess
        total_hess_w_reg = (total_hess_w_reg + total_hess_w_reg.T) / 2.

        if check_pos_def:
            pd_flag = self.check_pos_def(total_hess_w_reg)
            if not pd_flag:
                print("Converting hessian to nearest positive-definite matrix")
                total_hess_w_reg = nearest_pd(total_hess_w_reg)
                self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def log_loss(self, x, y, sample_weight=None, l2_reg=False, eps=1e-16) -> float:
        """ Compute binary cross-entropy loss """

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        pred, _ = self.pred(x)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            weight = self.model.weight.flatten()
            weight = weight.detach().cpu().numpy()
            log_loss += self.l2_reg * np.linalg.norm(weight, ord=2) / 2.

        return log_loss

    def log_loss_tensor(self, y: Tensor, pred: Tensor, sample_weight: Tensor = None, l2_reg=False, eps=1e-16) -> Tensor:
        log_loss = - y * torch.log(pred + eps) - (1. - y) * torch.log(1. - pred + eps)
        try:
            log_loss = torch.matmul(sample_weight, log_loss)
        except:
            log_loss = sample_weight.mul(log_loss)
        if l2_reg:
            weight = self.model.weight
            log_loss += torch.linalg.norm(weight, ord=2).div(2.).mul(self.l2_reg)

        return log_loss


class NNLastLayerIF(IFBaseClass):
    """
    Neural Networks
    Currently only support binary classification and training in GPU
    Only compute the influence for the last linear layer
    """

    def __init__(
            self,
            input_dim: int,
            l2_reg: float,
            base_model_cls: nn.Module,
            n_iter: int = 10000,
            lr: float = 1e-2,
            device: str = "cuda:0",
    ):
        self.input_dim = input_dim
        self.l2_reg = l2_reg
        self.base_model_cls = base_model_cls
        self.n_iter = n_iter
        self.lr = lr
        self.input_dim = input_dim
        self.device = torch.device(device)

        self.logistic = sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=(1. / l2_reg),
            fit_intercept=False,
            tol=1e-4,
            solver="newton-cg",
            max_iter=2048,
            multi_class="ovr",
            warm_start=False,
        )

    def fit(self, x, y, sample_weight=None, save_path: str = None):
        """ Currently using full batch training """

        self.model = self.base_model_cls(self.input_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=self.l2_reg)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=self.l2_reg)

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        x_tensor = torch.from_numpy(x).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        sample_weight_tensor = torch.from_numpy(sample_weight).to(self.device)

        criterion = nn.BCELoss(sample_weight_tensor)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.n_iter // 4, gamma=0.1)

        self.model.train()
        for _ in tqdm(range(self.n_iter), desc="Training"):
            self.optimizer.zero_grad()
            pred = self.model(x_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        self.model.eval()

        """ Use logistic regression to fit the last layer """

        emb = self.emb(x)
        self.logistic.fit(emb, y, sample_weight=sample_weight)
        self.last_fc_weight = self.logistic.coef_.flatten()

        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

        return

    def load_model(self, path: str) -> None:
        self.model = self.base_model_cls(self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(path))
        return

    def pred(self, x):
        emb = self.emb(x)
        pred, pred_label = self.logistic.predict_proba(emb)[:, 1], self.logistic.predict(emb)
        return pred, pred_label

    def emb(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            emb = self.model.emb(x)
        emb = emb.cpu().numpy().astype(np.float64)

        return emb

    def retrain_last_fc(self, x: np.ndarray, y: np.ndarray,
                        sample_weight: np.ndarray or Sequence[float] = None) -> None:
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        emb = self.emb(x)
        self.logistic.fit(emb, y, sample_weight=sample_weight)

        return

    def log_loss(self, x, y, sample_weight=None, l2_reg=False, eps=1e-16):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        pred, _, = self.pred(x)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            log_loss += self.l2_reg * np.linalg.norm(self.last_fc_weight, ord=2) / 2.

        return log_loss

    def grad(self, x, y, sample_weight=None, l2_reg=False):
        """
        Compute the gradient for the last layer: grad_wo_reg = (pred - y) * x
        """

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        emb = self.emb(x)
        pred, _ = self.pred(x)

        indiv_grad = emb * (pred - y).reshape(-1, 1)
        reg_grad = self.l2_reg * self.last_fc_weight
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)

        total_grad = np.sum(weighted_indiv_grad, axis=0)
        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)
        emb = self.emb(x)
        indiv_grad = emb * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

        return total_grad, weighted_indiv_grad

    def hess(self, x, sample_weight=None, check_pos_def=False) -> np.ndarray:
        """
        Compute hessian matrix for the last layer: hessian = pred * (1 - pred) @ x^T @ x + lambda
        """

        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        emb = self.emb(x)
        pred, _ = self.pred(x)

        factor = pred * (1. - pred)
        indiv_hess = np.einsum("a,ai,aj->aij", factor, emb, emb)
        reg_hess = self.l2_reg * np.eye(emb.shape[1])
        hess_wo_reg = np.einsum("aij,a->ij", indiv_hess, sample_weight)
        total_hess_w_reg = hess_wo_reg + reg_hess

        if check_pos_def:
            self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg


if __name__ == "__main__":
    data = fetch_data("german")

    model = LogisticRegression(l2_reg=data.l2_reg)
    # model = NN(input_dim=data.dim, l2_reg=1e-3)
    # model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-3)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    model.fit(data.x_train, data.y_train)
    log_loss = model.log_loss(data.x_train, data.y_train)
    print("Training loss: ", log_loss)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)
    val_evaluator(data.y_val, pred_label_val)
    test_evaluator(data.y_test, pred_label_test)

    # total_grad, weighted_indiv_grad = model.grad(data.x_train, data.y_train)
    # print(total_grad.shape)
    # print(weighted_indiv_grad.shape)
    # hess = model.hess(data.x_train, data.y_train, check_pos_def=True)
    # total_grad, weighted_indiv_grad = model.grad(data.x_train, data.y_train)
    # print(hess.shape)
