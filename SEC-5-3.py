import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from dataset2 import fetch_data2, DataTemplate2
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv

import json

from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn, calc_robust_acc_nn

import pickle
import random

import copy

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import seaborn as sns
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/robust")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=500, help="points to delete")
    parser.add_argument('--random_seed', type=int, default=42, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")

    args = parser.parse_args()

    return args




def get_full_dataset(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()

    data: DataTemplate = fetch_data(args.dataset)
    return data


def train_model(args, data):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    """ vanilla training """

    model.fit(data.x_train, data.y_train)

    return model


def compute_influence(args, data, model):
    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError

    if args.model_type != 'nn':
        robust_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
    else:
        robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    
    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return util_pred_infl



def find_trimming_points(args, I2):

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    return indices_to_delete



def delete_points(args, data, points_idx):
    trimmed_data = copy.deepcopy(data)

    X, y, s = [], [], []

    for idx in range(trimmed_data.x_train.shape[0]):
        if idx in points_idx:
            continue
        X.append(trimmed_data.x_train[idx])
        y.append(trimmed_data.y_train[idx])
        s.append(trimmed_data.s_train[idx])

    trimmed_data.x_train = np.array(X)
    trimmed_data.y_train = np.array(y)
    trimmed_data.s_train = np.array(s)
    return trimmed_data



def attack_lr(args, data, model):
    clf = model.model

    num2attack = np.random.randint(int(0.05*data.x_test.shape[0]), int(0.25*data.x_test.shape[0]))
    idx2attack = np.random.choice(data.x_test.shape[0], num2attack, replace=False)

    w = clf.coef_[0]
    b = clf.intercept_
    x_adv = []
    for i,x0 in enumerate(data.x_test):
        if i not in idx2attack:
            x_adv.append(x0)
            continue
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
        x1 = x0 - perturbation
        x_adv.append(x1)
    x_adv = np.array(x_adv)

    return x_adv




if __name__ == "__main__":
    args = parse_args()

    pre, post, defense = [], [], []
    n_iters = 10
    args.seed = None

    data = get_full_dataset(args)

    args.points_to_delete = int(0.025*len(data.x_train))
    model = train_model(args, data)

    for _ in range(n_iters):
        x_adv_test = attack_lr(args, data, model)

        data.x_val, data.y_val, data.s_val = x_adv_test, data.y_test, data.s_test


        influences = compute_influence(args, data, model)
        points_idx = find_trimming_points(args, influences)
        trimmed_data = delete_points(args, data, points_idx)
        defense_model = train_model(args, trimmed_data)


        pre.append(accuracy_score(model.pred(data.x_test)[1], data.y_test))
        post.append(accuracy_score(model.pred(x_adv_test)[1], data.y_test))
        defense.append(accuracy_score(defense_model.pred(x_adv_test)[1], data.y_test))


    print(pre, post, defense)
