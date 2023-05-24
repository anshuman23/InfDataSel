import numpy as np

from model import IFBaseClass


def grad_ferm(grad_fn: IFBaseClass.grad, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Fair empirical risk minimization for binary sensitive attribute
    Exp(L|grp_0) - Exp(L|grp_1)
    """

    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    grad_grp_0_y_1, _ = grad_fn(x=x[idx_grp_0_y_1], y=y[idx_grp_0_y_1])
    grad_grp_1_y_1, _ = grad_fn(x=x[idx_grp_1_y_1], y=y[idx_grp_1_y_1])

    return (grad_grp_0_y_1 / len(idx_grp_0_y_1)) - (grad_grp_1_y_1 / len(idx_grp_1_y_1))


def loss_ferm(loss_fn: IFBaseClass.log_loss, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:
    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    loss_grp_0_y_1 = loss_fn(x[idx_grp_0_y_1], y[idx_grp_0_y_1])
    loss_grp_1_y_1 = loss_fn(x[idx_grp_1_y_1], y[idx_grp_1_y_1])

    return (loss_grp_0_y_1 / len(idx_grp_0_y_1)) - (loss_grp_1_y_1 / len(idx_grp_1_y_1))


def grad_dp(grad_fn: IFBaseClass.grad_pred, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ Demographic parity """

    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    grad_grp_0, _ = grad_fn(x=x[idx_grp_0])
    grad_grp_1, _ = grad_fn(x=x[idx_grp_1])

    return (grad_grp_1 / len(idx_grp_1)) - (grad_grp_0 / len(idx_grp_0))


def loss_dp(x: np.ndarray, s: np.ndarray, pred: np.ndarray) -> float:
    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    pred_grp_0 = np.sum(pred[idx_grp_0])
    pred_grp_1 = np.sum(pred[idx_grp_1])

    return (pred_grp_1 / len(idx_grp_1)) - (pred_grp_0 / len(idx_grp_0))
