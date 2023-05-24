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

import time

import dtreeviz

from sklearn.tree import DecisionTreeRegressor

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="dp", help="eop or dp")
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

    #print(data.x_train.shape, data.y_train.shape)

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
    robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return util_pred_infl, fair_pred_infl, robust_pred_infl






if __name__ == "__main__":
    args = parse_args()

    args.seed = 42

    data = get_full_dataset(args)

    model = train_model(args, data)

    start_time = time.time()
    util_inf, fair_inf, robust_inf = compute_influence(args, data, model)
    end_time = time.time()

    print("Time taken for influence computation : {}".format(end_time-start_time))


    ######################################
    X = np.append(data.x_train, data.y_train.reshape(-1,1), 1)
    y = copy.deepcopy(util_inf)
    y /= np.max(np.abs(y),axis=0)

    reg = DecisionTreeRegressor(random_state=42, max_depth=3)

    reg.fit(X, y)

    col_names = ['Feature {}'.format(i+1) for i in range(X.shape[1]-1)]
    col_names += ['Class Label']

    viz = dtreeviz.model(model=reg,
               X_train=X,
               y_train=y,
               target_name='Utility Influence',
               feature_names=col_names)

    v = viz.view()
    #v.save("tree_viz/"+args.dataset+"-utility.svg")
    ######################################


    ######################################
    X = np.append(data.x_train, data.y_train.reshape(-1,1), 1)
    y = copy.deepcopy(fair_inf)
    y /= np.max(np.abs(y),axis=0)

    reg = DecisionTreeRegressor(random_state=42, max_depth=3)

    reg.fit(X, y)

    col_names = ['Feature {}'.format(i+1) for i in range(X.shape[1]-1)]
    col_names += ['Class Label']

    viz = dtreeviz.model(model=reg,
               X_train=X,
               y_train=y,
               target_name='Fairness Influence',
               feature_names=col_names)

    v = viz.view()
    #v.save("tree_viz/"+args.dataset+"-fair.svg")
    ######################################


    ######################################
    X = np.append(data.x_train, data.y_train.reshape(-1,1), 1)
    y = copy.deepcopy(robust_inf)
    y /= np.max(np.abs(y),axis=0)

    reg = DecisionTreeRegressor(random_state=42, max_depth=3)

    reg.fit(X, y)

    col_names = ['Feature {}'.format(i+1) for i in range(X.shape[1]-1)]
    col_names += ['Class Label']

    viz = dtreeviz.model(model=reg,
               X_train=X,
               y_train=y,
               target_name='Robustness Influence',
               feature_names=col_names)

    v = viz.view()
    #v.save("tree_viz/"+args.dataset+"-robust.svg")
    ######################################

