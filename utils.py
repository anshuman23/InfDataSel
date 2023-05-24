# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os
import pandas as pd
import numpy as np
import random
from numpy import linalg as la
from typing import Dict
import torch


def save2csv(data: Dict, path: str) -> None:
    data = pd.DataFrame(data, index=[0])
    if not os.path.isfile(path):
        print("Creating new .csv file to save the results")
    else:
        print("Saving to existing .csv file %s" % path)
        est_res = pd.read_csv(path)
        data = pd.concat([est_res, data])

    data.to_csv(path, index=False)

    return


def fix_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return


def nearest_pd(M: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194

    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (M + M.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if check_positive_definite(A3):
        return A3

    spacing = np.spacing(la.norm(M))

    """
    The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    `spacing` will, for Gaussian random matrixes of small dimension, be on
    othe order of 1e-16. In practice, both ways converge, as the unit test
    below suggests.
    """

    I = np.eye(M.shape[0])
    k = 1
    while not check_positive_definite(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def check_positive_definite(M: np.ndarray) -> bool:
    """ Returns true when input is positive-definite, via Cholesky """

    try:
        _ = la.cholesky(M)
        return True
    except la.LinAlgError:
        return False
