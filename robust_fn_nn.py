import numpy as np
from sklearn.metrics import accuracy_score
import random

def grad_robust_nn(model, x_val, y_val):

    clf = model.logistic
    grad_fn = model.grad

    x_val = model.emb(x_val)

    w = clf.coef_[0]
    b = clf.intercept_
    x_val_adv = []
    for i,x0 in enumerate(x_val):
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
        x1 = x0 - perturbation
        x_val_adv.append(x1)
    x_val_adv = np.array(x_val_adv)
    

    #x_val_adv = np.load('xadv_val.npy')

    total_loss_grad, t = grad_fn(x=x_val_adv, y=y_val)
    return total_loss_grad


def calc_robust_acc_nn(model, x_val, y_val, type, stage):

    clf = model.logistic

    if stage == 'pre':
        x_val = model.emb(x_val)

        w = clf.coef_[0]
        b = clf.intercept_
        x_val_adv = []
        for i,x0 in enumerate(x_val):
            perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
            x1 = x0 - perturbation
            x_val_adv.append(x1)
        x_val_adv = np.array(x_val_adv)

        np.save('xadv_'+type+'.npy', x_val_adv)

    elif stage == 'post':
        x_val_adv = np.load('xadv_'+type+'.npy')

    y_val_adv = clf.predict(x_val_adv)
    return accuracy_score(y_val_adv, y_val)
