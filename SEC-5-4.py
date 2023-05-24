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


from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, default=42, help="random seed")
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


def run_surrogate_compute_influence(args, data):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.dataset == "toy" and args.save_model == "y":
        pickle.dump(model.model, open("toy/model.pkl", "wb"))

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

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)


    if args.model_type != 'nn':
        val_rob_acc = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
        test_rob_acc = calc_robust_acc(model, data.x_test, data.y_test, 'test', 'pre')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    else:
        val_rob_acc = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'pre')
        test_rob_acc = calc_robust_acc_nn(model, data.x_test, data.y_test, 'test', 'pre')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})


    return util_pred_infl




def find_trimming_points(args, I2):

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    return indices_to_delete


def delete_points(args, points_idx, batches):
    X, y, = [], []

    for idx in range(batches[0].shape[0]):
        if idx in points_idx:
            continue
        X.append(batches[0][idx])
        y.append(batches[1][idx])

    return np.array(X), np.array(y)


if __name__ == "__main__":
    args = parse_args()

    pre_dict, post_dict = {'LR':[], 'SVM':[]},  {'LR':[], 'SVM':[]}

    full_data = get_full_dataset(args)

    num_batches = 10
    bi = np.linspace(0, len(full_data.x_train), num=num_batches+1).astype(int)
    
    clf_lr = SGDClassifier(loss='log') # 'log' for LR, 'hinge' for linear SVM
    clf_svm = SGDClassifier(loss='hinge') # 'log' for LR, 'hinge' for linear SVM
    clf_pa = PassiveAggressiveClassifier()
    clf_lr_post = SGDClassifier(loss='log') # 'log' for LR, 'hinge' for linear SVM
    clf_svm_post = SGDClassifier(loss='hinge') # 'log' for LR, 'hinge' for linear SVM
    clf_pa_post = PassiveAggressiveClassifier()

    for i in range(num_batches):
        
        print("\nBatch Idx ==>", i+1)
        batches_x = full_data.x_train[bi[i]:bi[i+1]]
        batches_y = full_data.y_train[bi[i]:bi[i+1]]
        batches_s = full_data.s_train[bi[i]:bi[i+1]]
        

        #Flip some labels at random to create "bad" points
        idx_to_change = random.sample([i for i in range(batches_y.shape[0])], int(0.35*len(batches_x)))
        for j in idx_to_change:
            batches_y[j] = 1 - batches_y[j]

        data_copy = copy.deepcopy(full_data)
        data_copy.x_train = batches_x
        data_copy.y_train = batches_y
        data_copy.s_train = batches_s

        clf_lr.partial_fit(batches_x, batches_y, classes=[0,1])
        clf_svm.partial_fit(batches_x, batches_y, classes=[0,1])
        clf_pa.partial_fit(batches_x, batches_y, classes=[0,1])

        pre_dict['LR'].append(accuracy_score(clf_lr.predict(full_data.x_test), full_data.y_test))
        pre_dict['SVM'].append(accuracy_score(clf_svm.predict(full_data.x_test), full_data.y_test))


        args.points_to_delete = int(0.1*len(batches_x))

        influences = run_surrogate_compute_influence(args, data_copy) 
        deletion_points = find_trimming_points(args, influences)

        batches_x, batches_y = delete_points(args, deletion_points, (batches_x, batches_y))

        clf_lr_post.partial_fit(batches_x, batches_y, classes=[0,1])
        clf_svm_post.partial_fit(batches_x, batches_y, classes=[0,1])
        clf_pa_post.partial_fit(batches_x, batches_y, classes=[0,1])

        post_dict['LR'].append(accuracy_score(clf_lr_post.predict(full_data.x_test), full_data.y_test))
        post_dict['SVM'].append(accuracy_score(clf_svm_post.predict(full_data.x_test), full_data.y_test))


    print(pre_dict)
    print(post_dict)
