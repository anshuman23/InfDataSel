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



def pre_main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate = fetch_data(args.dataset)
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

    np.save('explainer/data/binaries/util_infl.npy', util_pred_infl)
    np.save('explainer/data/binaries/fair_infl.npy', fair_pred_infl)
    np.save('explainer/data/binaries/robust_infl.npy', robust_pred_infl)

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


    np.save('trn.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return val_res, test_res



def post_main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate2 = fetch_data2(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.dataset == "toy" and args.save_model == "y":
        pickle.dump(model.model, open("toy/model.pkl", "wb"))


    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)


    
    if args.model_type != 'nn':
        val_rob_acc = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    else:
        val_rob_acc = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc_nn(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})


    return val_res, test_res


def deletion_process(args):
    X_org = np.load('trn.npy')
    num_to_del = int(args.points_to_delete)
    num_features = X_org.shape[1]

    I = np.load('explainer/data/binaries/'+args.type+'_infl.npy')

    if args.strategy == 'inc':
        indices_to_delete = I.argsort()[::-1][-num_to_del:][::-1].tolist() #INC by deleting bad points
    elif args.strategy == 'dec':
        indices_to_delete = I.argsort()[-num_to_del:][::-1].tolist() #DEC by deleting good points
    elif args.strategy == 'random':
        random.seed(int(args.random_seed)) #42, 1, 35, 999, 5454
        indices_to_delete = random.sample(range(0,len(I)), int(0.1*len(I)))[:num_to_del] #RANDOM

    X_new = []
    for i in range(X_org.shape[0]):
        if i in indices_to_delete:
            continue
        X_new.append(X_org[i])
    X_new = np.array(X_new)


    X = X_new
    np.save('2trn.npy', X)

    return indices_to_delete


if __name__ == "__main__":
    args = parse_args()

    pre_val_res, pre_test_res = pre_main(args) #Run pre code

    influence_test_results = {'fair': [pre_test_res[args.metric]]} #Initialize results dict (infl)
    random_test_results = {42: copy.deepcopy(influence_test_results), #Initialize results dict (random)
        1: copy.deepcopy(influence_test_results),
        35: copy.deepcopy(influence_test_results),
        999: copy.deepcopy(influence_test_results),
        5454: copy.deepcopy(influence_test_results),
    }


    x_ax = np.linspace(0, int(0.05*len(np.load('explainer/data/binaries/'+args.type+'_infl.npy'))), num=11) #Set up x axis range according to data

    if args.only_pre == 'y': #If only pre needed, exit
        exit(0)

    args.strategy, args.type = 'inc', 'fair' #EOP, increase fairness
    for i,num_i in enumerate(x_ax[1:]):
        args.points_to_delete = int(num_i)

        fair_del_idx = deletion_process(args)
        post_val_res, post_test_res = post_main(args)

        influence_test_results['fair'].append(post_test_res[args.metric])


    args.strategy = 'random' #Random baseline experiments
    for seedval in [42,1,35,999,5454]:
        args.random_seed = seedval
        for i,num_i in enumerate(x_ax[1:]):
            args.points_to_delete = int(num_i)
            #print('-->>', args.points_to_delete, args.random_seed)
            deletion_process(args)
            post_val_res, post_test_res = post_main(args)

            random_test_results[args.random_seed]['fair'].append(post_test_res[args.metric])


    with open('appendix-d-outputs/influence-'+args.dataset+'-'+args.model_type+'.pkl', "wb") as output_file:
        pickle.dump(influence_test_results, output_file)
    with open('appendix-d-outputs/random-'+args.dataset+'-'+args.model_type+'.pkl', "wb") as output_file:
        pickle.dump(random_test_results, output_file)


    # Save influence values
    fair_infl_vals = np.load('explainer/data/binaries/fair_infl.npy')

    deleted_fair_influences = {'fair':fair_infl_vals[fair_del_idx]}

    with open('appendix-d-outputs/fair-'+args.dataset+'-'+args.model_type+'.pkl', "wb") as output_file:
        pickle.dump(deleted_fair_influences, output_file)
