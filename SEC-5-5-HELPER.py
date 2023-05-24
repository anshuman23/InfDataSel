import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from model import LogisticRegression, NNLastLayerIF, MLPClassifier
from utils import fix_seed, save2csv

import json

import pickle
import random

import copy

import pandas as pd

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument('--plot_before_only', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")
    parser.add_argument('--diabetes', type=str, default="y", help="y/n")

    args = parser.parse_args()

    return args



def plot_results_v2(n_rounds, savename='al.png'):
    x = [i for i in range(n_rounds+1)]

    with open("sec-5-5-outputs/dr_results_1.pkl", "rb") as input_file:
        data1 = pickle.load(input_file)
    with open("sec-5-5-outputs/dr_results_2.pkl", "rb") as input_file:
        data2 = pickle.load(input_file)
    with open("sec-5-5-outputs/dr_results_3.pkl", "rb") as input_file:
        data3 = pickle.load(input_file)
    with open("sec-5-5-outputs/dr_results_4.pkl", "rb") as input_file:
        data4 = pickle.load(input_file)
    with open("sec-5-5-outputs/dr_results_5.pkl", "rb") as input_file:
        data5 = pickle.load(input_file)


    key = 'random'
    rand = np.mean([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)
    rand_err = np.std([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)

    key = 'entropy'
    ent = np.mean([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)
    ent_err = np.std([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)

    key = 'margin'
    mar = np.mean([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)
    mar_err = np.std([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)

    key = 'uncertainty'
    unct = np.mean([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)
    unct_err = np.std([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)

    key = 'influence'
    inf = np.mean([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)
    inf_err = np.std([data1[key], data2[key], data3[key], data4[key], data5[key]], axis=0)


    plt.plot(x, rand, label='Random Sampling', color='tab:orange', linestyle='dashed',lw=3)
    plt.fill_between(x, rand-rand_err, rand+rand_err, color='tab:orange', alpha=0.2)

    plt.plot(x, ent, label='Entropy Sampling', color='tab:blue', linestyle='dashed',lw=3)
    plt.fill_between(x, ent-ent_err, ent+ent_err, color='tab:blue', alpha=0.2)

    plt.plot(x, mar, label='Margin Sampling', color='tab:green', linestyle='dashed',lw=3)
    plt.fill_between(x, mar-mar_err, mar+mar_err, color='tab:green', alpha=0.2)

    plt.plot(x, unct, label='Uncertainty Sampling', color='tab:purple', linestyle='dashed',lw=3)
    plt.fill_between(x, unct-unct_err, unct+unct_err, color='tab:purple', alpha=0.2)

    plt.plot(x, inf, label='Influence Sampling', color='tab:red', linestyle='solid',lw=3) 
    plt.fill_between(x, inf-inf_err, inf+inf_err, color='tab:red', alpha=0.2)

    plt.xlabel('# Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig(savename, bbox_inches='tight', dpi=300)
    plt.show()



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


def create_initial_and_pool(X, y, n_initial=100):
    initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
    X_training, y_training = X[initial_idx], y[initial_idx]

    pool_idx = list(set([i for i in range(len(X))]) - set(initial_idx))
    X_init_pool, y_init_pool = X[pool_idx], y[pool_idx]

    return X_training, y_training, X_init_pool, y_init_pool



def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def influence_sampling(classifier, X_pool):
    I = estimation_model.predict(X_pool)

    query_idx = np.argmax(I)

    return query_idx, X_pool[query_idx]



def train_model(args, x_train, y_train):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    model = LogisticRegression(l2_reg=1)

    """ vanilla training """

    model.fit(x_train, y_train)

    return model


def compute_influence(args, x_train, y_train, x_val, y_val, model):
    ori_util_loss_val = model.log_loss(x_val,y_val)

    """ compute the influence and save data """

    pred_train, _ = model.pred(x_train)

    train_total_grad, train_indiv_grad = model.grad(x_train, y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(x_val, y_val)

    hess = model.hess(x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)

    return util_pred_infl



if __name__ == "__main__":

    args = parse_args()

    #######################
    #args.plot_before_only = 'y'
    args.seed = None
    #args.dataset = 'toy'
    #args.diabetes = 'y'
    N_QUERIES = 10 #100
    N_ROUNDS = 10
    base_model = clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=None))
    #######################

    if args.plot_before_only == 'y':
        if args.dataset == 'toy': #Just a placeholder dataset name since diabetes has not been incorporated in the regular way
            plot_results_v2(N_ROUNDS, 'diabetes_mean_std')
        else:
            plot_results_v2(N_ROUNDS, args.dataset)
        exit(0)

    data = get_full_dataset(args)

    #X_init, y_init, X_pool, y_pool = create_initial_and_pool(data.x_train, data.y_train, n_initial=300) #Not reliable, use deterministic splits instead
    if args.diabetes == 'n':
        X_init, y_init, X_pool, y_pool = np.load('data/al-data/X_init_'+args.dataset+'.npy'), np.load('data/al-data/y_init_'+args.dataset+'.npy'), np.load('data/al-data/X_pool_'+args.dataset+'.npy'), np.load('data/al-data/y_pool_'+args.dataset+'.npy')

    with open("data/al-data/diabetic_retinopathy.pkl", "rb") as input_file:
        datadict = pickle.load(input_file)
    if args.diabetes == 'y':
        X_init, y_init, X_pool, y_pool, data.x_val, data.y_val, data.x_test, data.y_test = datadict['X_init'], datadict['y_init'], datadict['X_pool'], datadict['y_pool'], datadict['X_val'], datadict['y_val'], datadict['X_test'], datadict['y_test']


    X_pool_copy, y_pool_copy = copy.deepcopy(X_pool), copy.deepcopy(y_pool)

    surrogate_model = train_model(args, X_init, y_init)
    util_infl = compute_influence(args, X_init, y_init, data.x_val, data.y_val, surrogate_model)
    estimation_model = DecisionTreeRegressor(random_state=42).fit(X_init, util_infl)

    #Random Sampling
    rlearner = ActiveLearner(
    estimator=base_model,
    query_strategy=random_sampling,
    X_training=X_init, y_training=y_init)

    unqueried_score = rlearner.score(data.x_test, data.y_test)
    results_dict = {'random':[unqueried_score], 'entropy':[unqueried_score], 'margin':[unqueried_score], 'influence':[unqueried_score], 'uncertainty':[unqueried_score], 'influence':[unqueried_score]}

    for _ in range(N_ROUNDS):
        for index in range(N_QUERIES):
            query_index, query_instance = rlearner.query(X_pool)

            x_teach, y_teach = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            rlearner.teach(X=x_teach, y=y_teach)

            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        rmodel_accuracy = rlearner.score(data.x_test, data.y_test)
        results_dict['random'].append(rmodel_accuracy)

    X_pool, y_pool = copy.deepcopy(X_pool_copy), copy.deepcopy(y_pool_copy)

    print("RANDOM SAMPLING DONE.")


    #Entropy Sampling
    elearner = ActiveLearner(
    estimator=base_model,
    query_strategy=entropy_sampling,
    X_training=X_init, y_training=y_init)

    for _ in range(N_ROUNDS):
        for index in range(N_QUERIES):
            query_index, query_instance = elearner.query(X_pool)

            x_teach, y_teach = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            elearner.teach(X=x_teach, y=y_teach)

            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        emodel_accuracy = elearner.score(data.x_test, data.y_test)
        results_dict['entropy'].append(emodel_accuracy)

    X_pool, y_pool = copy.deepcopy(X_pool_copy), copy.deepcopy(y_pool_copy)

    print("ENTROPY SAMPLING DONE.")


    #Margin Sampling
    mlearner = ActiveLearner(
    estimator=base_model,
    query_strategy=margin_sampling,
    X_training=X_init, y_training=y_init)

    for _ in range(N_ROUNDS):
        for index in range(N_QUERIES):
            query_index, query_instance = mlearner.query(X_pool)

            x_teach, y_teach = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            mlearner.teach(X=x_teach, y=y_teach)

            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        mmodel_accuracy = mlearner.score(data.x_test, data.y_test)
        results_dict['margin'].append(mmodel_accuracy)

    X_pool, y_pool = copy.deepcopy(X_pool_copy), copy.deepcopy(y_pool_copy)

    print("MARGIN SAMPLING DONE.")


    #Uncertainty Sampling
    ulearner = ActiveLearner(
    estimator=base_model,
    query_strategy=uncertainty_sampling,
    X_training=X_init, y_training=y_init)

    for _ in range(N_ROUNDS):
        for index in range(N_QUERIES):
            query_index, query_instance = ulearner.query(X_pool)

            x_teach, y_teach = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            ulearner.teach(X=x_teach, y=y_teach)

            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        umodel_accuracy = ulearner.score(data.x_test, data.y_test)
        results_dict['uncertainty'].append(umodel_accuracy)

    X_pool, y_pool = copy.deepcopy(X_pool_copy), copy.deepcopy(y_pool_copy)

    print("UNCERTAINTY SAMPLING DONE.")


    #Influence Sampling
    ilearner = ActiveLearner(
    estimator=base_model,
    query_strategy=influence_sampling,
    X_training=X_init, y_training=y_init)

    for _ in range(N_ROUNDS):
        for index in range(N_QUERIES):
            query_index, query_instance = ilearner.query(X_pool)

            x_teach, y_teach = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
            ilearner.teach(X=x_teach, y=y_teach)

            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        imodel_accuracy = ilearner.score(data.x_test, data.y_test)
        results_dict['influence'].append(imodel_accuracy)

    X_pool, y_pool = copy.deepcopy(X_pool_copy), copy.deepcopy(y_pool_copy)

    print("INFLUENCE SAMPLING DONE.")

    #np.save('al-data/X_init_'+args.dataset+'.npy', X_init)
    #np.save('al-data/X_pool_'+args.dataset+'.npy', X_pool)
    #np.save('al-data/y_init_'+args.dataset+'.npy', y_init)
    #np.save('al-data/y_pool_'+args.dataset+'.npy', y_pool)

    #print(results_dict)

    with open("sec-5-5-outputs/dr_results.pkl", "wb") as output_file:
        pickle.dump(results_dict, output_file)
