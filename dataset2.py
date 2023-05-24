import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Mapping, Optional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class DataTemplate2():
    def __init__(self, x_train, y_train, s_train, x_val, y_val, s_val, x_test, y_test, s_test, l2_reg, s_col_idx):
        self.num_train: int = x_train.shape[0]
        self.num_val: int = x_val.shape[0]
        self.num_test: int = x_test.shape[0]
        self.dim: int = x_train.shape[1]
        self.num_s_feat = len(Counter(s_train))
        self.l2_reg = l2_reg
        self.s_col_idx = s_col_idx

        self.x_train: np.ndarray = x_train
        self.y_train: np.ndarray = y_train
        self.s_train: np.ndarray = s_train
        self.x_val: np.ndarray = x_val
        self.y_val: np.ndarray = y_val
        self.s_val: np.ndarray = s_val
        self.x_test: np.ndarray = x_test
        self.y_test: np.ndarray = y_test
        self.s_test: np.ndarray = s_test

        print("Dataset statistic - #total: %d; #train: %d; #val.: %d; #test: %d; #dim.: %.d\n"
              % (self.num_train + self.num_val + self.num_test,
                 self.num_train, self.num_val, self.num_test, self.dim))


class Dataset():
    """
    General dataset
    Assure in a binary group case, Grp. 1 is the privileged group and Grp. 0 is the unprivileged group
    Assure in a binary label case, 1. is the positive outcome and 0. is the negative outcome
    Sensitive feature is not excluded from data
    """

    def __init__(self, name, df, target_feat, sensitive_feat, l2_reg, test_df=None, categorical_feat=None,
                 drop_feat=None, s_thr=None, label_mapping=None, shuffle=False, load_idx=True, idx_path=None,
                 test_p=0.20, val_p=0.25, *args, **kwargs):
        """

        :param name: dataset name
        :param df: dataset DataFrame
        :param target_feat: feature to be predicted
        :param sensitive_feat: sensitive feature
        :param l2_reg: strength of l2 regularization for logistic regression model
        :param test_df: DataFrame for testing, optional
        :param categorical_feat: categorical features to be processed into one-hot encoding
        :param drop_feat: features to drop
        :param s_thr: threshold to split the data into two group, only for continuous sensitive feature
        :param label_mapping: mapping for one-hot encoding for some features
        :param shuffle: shuffle the dataset
        :param load_idx: loading shuffled row index
        :param idx_path: path for the shuffled index file
        :param test_p: proportion of test data
        :param val_p: proportion of validation data
        """

        print("Loading %s dataset.." % name)

        self.categorical_feat = categorical_feat if categorical_feat is not None else []
        self.l2_reg = l2_reg

        if shuffle:
            if load_idx and os.path.exists(idx_path):
                with open(idx_path) as f:
                    shuffle_idx = json.load(f)
            else:
                shuffle_idx = np.random.permutation(df.index)
                with open(idx_path, "w") as f:
                    json.dump(shuffle_idx.tolist(), f)

            df = df.reindex(shuffle_idx)

        df.dropna(inplace=True)
        if drop_feat is not None:
            df.drop(columns=drop_feat, inplace=True)

        if test_df is None:
            num_test = round(len(df) * test_p)
            num_train_val = len(df) - num_test
            train_val_df = df.iloc[:num_train_val]
            test_df = df.iloc[num_train_val:]
        else:
            test_df.dropna(inplace=True)
            if drop_feat is not None:
                test_df.drop(columns=drop_feat, inplace=True)
            train_val_df = df

        s_train_val, s_test = train_val_df[sensitive_feat].to_numpy(), test_df[sensitive_feat].to_numpy()
        if s_thr is not None:
            s_train_val = np.where(s_train_val >= s_thr[0], s_thr[1]["larger"], s_thr[1]["smaller"])
            s_test = np.where(s_test > s_thr[0], s_thr[1]["larger"], s_thr[1]["smaller"])
        else:
            assert sensitive_feat in label_mapping
            s_train_val = np.array([label_mapping[sensitive_feat][e] for e in s_train_val])
            s_test = np.array([label_mapping[sensitive_feat][e] for e in s_test])

        train_val_df, updated_label_mapping = self.one_hot(train_val_df, label_mapping)
        test_df, _ = self.one_hot(test_df, updated_label_mapping)

        y_train_val, y_test = train_val_df[target_feat].to_numpy(), test_df[target_feat].to_numpy()
        train_val_df, test_df = train_val_df.drop(columns=target_feat), test_df.drop(columns=target_feat)

        num_val = round(len(train_val_df) * val_p)
        num_train = len(train_val_df) - num_val
        x_train, x_val = train_val_df.iloc[:num_train], train_val_df.iloc[num_train:]
        self.y_train, self.y_val = y_train_val[:num_train], y_train_val[num_train:]
        self.s_train, self.s_val = s_train_val[:num_train], s_train_val[num_train:]
        self.y_test, self.s_test = y_test, s_test

        self.x_train, scaler = self.center(x_train)
        self.x_val, _ = self.center(x_val, scaler)
        self.x_test, _ = self.center(test_df, scaler)

        self.s_col_idx = train_val_df.columns.tolist().index(sensitive_feat)

        if name.startswith("DistShiftAdult") or name == "Adult" or name == "German" or name == "Synthetic" or name == 'ACS' or name == 'Bank' or name =='Credit' or name =='CelebA' or name =='NLP':
            #Reconstituting training set by adding validation set back to training set 
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            ##print(self.x_train.shape, self.y_train.shape, self.s_train.shape)
            #print(self.x_train.shape, self.y_train.shape)


            #Divide test set into validation and test set
            self.x_test, self.x_val, self.y_test, self.y_val, self.s_test, self.s_val = train_test_split(self.x_test, self.y_test, self.s_test, test_size=0.5, random_state=42000, shuffle=True)

            #self.x_test, self.x_val = self.x_val, self.x_test
            #self.y_test, self.y_val = self.y_val, self.y_test
            #self.s_test, self.s_val = self.s_val, self.s_test

            t = np.load('2trn.npy')
            #print(t.shape)
            self.x_train, self.y_train = t[:,:t.shape[1]-1], t[:,t.shape[1]-1:].reshape((-1))
            #self.x_train, self.y_train = t[:,:102], t[:,102:].reshape((-1))
            #self.x_train, self.y_train = t[:,:56], t[:,56:].reshape((-1))


            #np.save('temp-bins/xz_train.npy', self.x_train)
            #np.save('temp-bins/y_train.npy', self.y_train)
            #np.save('temp-bins/z_train.npy', self.s_train)
            #np.save('temp-bins/xz_test.npy', self.x_test)
            #np.save('temp-bins/y_test.npy', self.y_test)
            #np.save('temp-bins/z_test.npy', self.s_test)
            #np.save('temp-bins/xz_val.npy', self.x_val)
            #np.save('temp-bins/y_val.npy', self.y_val)
            #np.save('temp-bins/z_val.npy', self.s_val)


            #print(self.x_train.shape, self.y_train.shape)
            #print(updated_label_mapping)

            ##print(self.x_test.shape, self.x_val.shape)


        if name.startswith('RAA_Attack') or name.startswith('NRAA_Attack') or name.startswith('IAF_Attack') or name.startswith('Solans_Attack'):
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            
            self.x_test, self.y_test, self.s_test, self.x_val, self.y_val, self.s_val  = self.x_test, self.y_test, self.s_test, self.x_test, self.y_test, self.s_test

            t = np.load('2trn.npy')
            self.x_train, self.y_train = t[:,:t.shape[1]-1], t[:,t.shape[1]-1:].reshape((-1))


        if name == "Toy":
            #Removing sensitive attribute for training
            self.x_train, self.x_val, self.x_test = self.x_train[:,:-1], self.x_val[:,:-1], self.x_test[:,:-1]

            #Reconstituting training set by adding validation set back to training set 
            self.x_train = np.vstack((self.x_train, self.x_val))
            self.y_train = np.hstack((self.y_train, self.y_val))
            self.s_train = np.hstack((self.s_train, self.s_val))
            ##print(self.x_train.shape, self.y_train.shape, self.s_train.shape)

            #Divide test set into validation and test set
            ##self.x_test, self.x_val, self.y_test, self.y_val, self.s_test, self.s_val = train_test_split(self.x_test, self.y_test, self.s_test, test_size=0.6, random_state=42, shuffle=False)
            self.x_val = self.x_test
            self.y_val = self.y_test
            self.s_val = self.s_test

            #self.x_test, self.x_val = self.x_val, self.x_test
            #self.y_test, self.y_val = self.y_val, self.y_test
            #self.s_test, self.s_val = self.s_val, self.s_test

            ##print(self.x_test.shape, self.x_val.shape)

            t = np.load('2trn.npy')
            #print(t.shape)
            self.x_train, self.y_train = t[:,:t.shape[1]-1], t[:,t.shape[1]-1:].reshape((-1))



    def one_hot(self, df: pd.DataFrame, label_mapping: Optional[Mapping]) -> Tuple[pd.DataFrame, Mapping]:
        label_mapping = {} if label_mapping is None else label_mapping
        updated_label_mapping = {}
        for c in df.columns:
            if c in self.categorical_feat:
                column = df[c]
                df = df.drop(c, axis=1)

                if c in label_mapping:
                    mapping = label_mapping[c]
                else:
                    unique_values = list(dict.fromkeys(column))
                    mapping = {v: i for i, v in enumerate(unique_values)}
                    updated_label_mapping[c] = mapping

                n = len(mapping)
                if n > 2:
                    for i in range(n):
                        col_name = '{}.{}'.format(c, i)
                        col_i = [1. if list(mapping.keys())[i] == e else 0. for e in column]
                        df[col_name] = col_i
                else:
                    col = [mapping[e] for e in column]
                    df[c] = col

        updated_label_mapping.update(label_mapping)

        return df, updated_label_mapping

    @staticmethod
    def center(X: pd.DataFrame, scaler: preprocessing.StandardScaler = None) -> Tuple:
        if scaler is None:
            scaler = preprocessing.StandardScaler().fit(X.values)
        scaled = scaler.transform(X.values)

        return scaled, scaler

    @property
    def data(self):
        return DataTemplate2(self.x_train, self.y_train, self.s_train,
                            self.x_val, self.y_val, self.s_val,
                            self.x_test, self.y_test, self.s_test,self.l2_reg, self.s_col_idx)

class NRAA_Attack_German(Dataset):

    def __init__(self):
        meta = json.load(open("./data/nraa_attack_german/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(NRAA_Attack_German, self).__init__(name="NRAA_Attack_German", df=train, test_df=test, **meta, shuffle=False)


class RAA_Attack_German(Dataset):

    def __init__(self):
        meta = json.load(open("./data/raa_attack_german/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(RAA_Attack_German, self).__init__(name="RAA_Attack_German", df=train, test_df=test, **meta, shuffle=False)                            


class Solans_Attack_Drug(Dataset):

    def __init__(self):
        meta = json.load(open("./data/solans_attack_drug/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Solans_Attack_Drug, self).__init__(name="Solans_Attack_Drug", df=train, test_df=test, **meta, shuffle=False)


class IAF_Attack_Drug(Dataset):

    def __init__(self):
        meta = json.load(open("./data/iaf_attack_drug/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(IAF_Attack_Drug, self).__init__(name="IAF_Attack_Drug", df=train, test_df=test, **meta, shuffle=False)


class NRAA_Attack_Drug(Dataset):

    def __init__(self):
        meta = json.load(open("./data/nraa_attack_drug/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(NRAA_Attack_Drug, self).__init__(name="NRAA_Attack_Drug", df=train, test_df=test, **meta, shuffle=False)



class RAA_Attack_Drug(Dataset):

    def __init__(self):
        meta = json.load(open("./data/raa_attack_drug/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(RAA_Attack_Drug, self).__init__(name="RAA_Attack_Drug", df=train, test_df=test, **meta, shuffle=False)



class Solans_Attack_Compas(Dataset):

    def __init__(self):
        meta = json.load(open("./data/solans_attack_compas/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Solans_Attack_Compas, self).__init__(name="Solans_Attack_Compas", df=train, test_df=test, **meta, shuffle=False)


class IAF_Attack_Compas(Dataset):

    def __init__(self):
        meta = json.load(open("./data/iaf_attack_compas/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(IAF_Attack_Compas, self).__init__(name="IAF_Attack_Compas", df=train, test_df=test, **meta, shuffle=False)


class NRAA_Attack_Compas(Dataset):

    def __init__(self):
        meta = json.load(open("./data/nraa_attack_compas/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(NRAA_Attack_Compas, self).__init__(name="NRAA_Attack_Compas", df=train, test_df=test, **meta, shuffle=False)


class RAA_Attack_Compas(Dataset):

    def __init__(self):
        meta = json.load(open("./data/raa_attack_compas/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(RAA_Attack_Compas, self).__init__(name="RAA_Attack_Compas", df=train, test_df=test, **meta, shuffle=False)






class Synthetic(Dataset):

    def __init__(self):
        meta = json.load(open("./data/synthetic/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Synthetic, self).__init__(name="Synthetic", df=train, test_df=test, **meta, shuffle=False)



class NLP(Dataset):

    def __init__(self):
        meta = json.load(open("./data/nlp/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(NLP, self).__init__(name="NLP", df=train, test_df=test, **meta, shuffle=False)

class CelebA(Dataset):

    def __init__(self):
        meta = json.load(open("./data/celeba/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(CelebA, self).__init__(name="CelebA", df=train, test_df=test, **meta, shuffle=False)



class Bank(Dataset):

    def __init__(self):
        meta = json.load(open("./data/bank/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Bank, self).__init__(name="Bank", df=train, test_df=test, **meta, shuffle=False)


class Credit(Dataset):

    def __init__(self):
        meta = json.load(open("./data/credit/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        #column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], skipinitialspace=True)
        test = pd.read_csv(meta["test_path"],  skipinitialspace=True)

        
        super(Credit, self).__init__(name="Credit", df=train, test_df=test, **meta, shuffle=False)


class AdultDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/adult """

    def __init__(self):
        meta = json.load(open("./data/adult/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"], header=0, names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        # remove the "." at the end of each "income"
        test["income"] = [e[:-1] for e in test["income"].values]

        super(AdultDataset, self).__init__(name="Adult", df=train, test_df=test, **meta, shuffle=False)



class ACSDataset(Dataset):

    def __init__(self):
        meta = json.load(open("./data/acs/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"],  names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        super(ACSDataset, self).__init__(name="ACS", df=train, test_df=test, **meta, shuffle=False)



class DistShiftAdult(Dataset):

    def __init__(self):
        meta = json.load(open("./data/dist_shift_adult/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"],  names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        
        super(DistShiftAdult, self).__init__(name="DistShiftAdult", df=train, test_df=test, **meta, shuffle=False)



class DistShiftAdultTime(Dataset):

    def __init__(self):
        meta = json.load(open("./data/dist_shift_adult_time/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"],  names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        
        super(DistShiftAdultTime, self).__init__(name="DistShiftAdultTime", df=train, test_df=test, **meta, shuffle=False)



class DistShiftAdultLoc(Dataset):

    def __init__(self):
        meta = json.load(open("./data/dist_shift_adult_loc/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"],  names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])

        
        super(DistShiftAdultLoc, self).__init__(name="DistShiftAdultLoc", df=train, test_df=test, **meta, shuffle=False)





class Toy(Dataset):

    def __init__(self):
        meta = json.load(open("./data/toy/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        column_names = meta["column_names"].split(",")
        train = pd.read_csv(meta["train_path"], names=column_names, skipinitialspace=True,
                            na_values=meta["na_values"])
        test = pd.read_csv(meta["test_path"], names=column_names, skipinitialspace=True,
                           na_values=meta["na_values"])        


        super(Toy, self).__init__(name="Toy", df=train, test_df=test, **meta, shuffle=False)



class CommDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/communities+and+crime """

    def __init__(self):
        meta = json.load(open("./data/communities/meta.json"))

        df = pd.read_csv(meta["train_path"], index_col=0)
        # invert the label to make 1. as the positive outcome
        df[meta["target_feat"]] = -(df[meta["target_feat"]].values - 1.)

        super(CommDataset, self).__init__(name="Comm", df=df, **meta, shuffle=False)


class CompasDataset(Dataset):
    """ https://github.com/propublica/compas-analysis """

    def __init__(self):
        meta = json.load(open("./data/compas/meta.json"))

        df = pd.read_csv(meta["train_path"], index_col='id')
        df = self.default_preprocessing(df)
        df = df[meta["features_to_keep"].split(",")]

        super(CompasDataset, self).__init__(name="Compas", df=df, **meta, shuffle=False)

    @staticmethod
    def default_preprocessing(df):
        """
        Perform the same preprocessing as the original analysis:
        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        def race(row):
            return 'Caucasian' if row['race'] == "Caucasian" else 'Not Caucasian'

        def two_year_recid(row):
            return 'Did recid.' if row['two_year_recid'] == 1 else 'No recid.'

        df['race'] = df.apply(lambda row: race(row), axis=1)
        df['two_year_recid'] = df.apply(lambda row: two_year_recid(row), axis=1)

        return df[(df.days_b_screening_arrest <= 30)
                  & (df.days_b_screening_arrest >= -30)
                  & (df.is_recid != -1)
                  & (df.c_charge_degree != 'O')
                  & (df.score_text != 'N/A')]

'''
class BankDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/bank+marketing """

    def __init__(self):
        meta = json.load(open("./data/bank/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        df = pd.read_csv(meta["train_path"], sep=";", na_values=meta["na_values"])

        super(BankDataset, self).__init__(name="Bank", df=df, **meta, shuffle=True)
'''

class GermanDataset(Dataset):
    """ https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 """

    def __init__(self):
        meta = json.load(open("./data/german/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")

        df = pd.read_csv(meta["train_path"], sep=" ", names=meta["column_names"].split(","))
        df = self.default_preprocessing(df)
        df["credit"] = df["credit"].astype("str")

        super(GermanDataset, self).__init__(name="German", df=df, **meta, shuffle=False)

    @staticmethod
    def default_preprocessing(df):
        """
        Adds a derived sex attribute based on personal_status.
        https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/german_dataset.py
        """

        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                      'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)

        return df


class MEPSDataset(Dataset):
    """ Borrowed from https://github.com/Trusted-AI/AIF360/tree/master/aif360/datasets """

    features_to_keep = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                        'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                        'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                        'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                        'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PCS42',
                        'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION']

    def __init__(self, panel: int, fy: str, df: pd.DataFrame, **kwargs):
        assert panel in (19, 20, 21)
        assert fy in ("15", "16")
        self.panel = panel
        self.fy = fy

        features_to_keep = MEPSDataset.features_to_keep.copy()
        features_to_keep.append('PERWT' + self.fy + 'F')

        df = self.default_preprocessing(df)
        df = df[features_to_keep]

        super(MEPSDataset, self).__init__(name="MEPS%d" % self.panel, df=df, **kwargs)

    def default_preprocessing(self, df):
        def race(row):
            if ((row['HISPANX'] == 2) and (
                    row['RACEV2X'] == 1)):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df = df.rename(columns={'RACEV2X': 'RACE'})

        df = df[df['PANEL'] == self.panel]

        # RENAME COLUMNS
        df = df.rename(columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                                'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                                'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                                'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                                'POVCAT' + self.fy: 'POVCAT', 'INSCOV' + self.fy: 'INSCOV'})

        df = df[df['REGION'] >= 0]  # remove values -1
        df = df[df['AGE'] >= 0]  # remove values -1

        df = df[df['MARRY'] >= 0]  # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9

        # for all other categorical features, remove values < -1
        df = df[(df[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                     'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                     'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                     'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                     'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]

        def utilization(row):
            return row['OBTOTV' + self.fy] + row['OPTOTV' + self.fy] + row['ERTOT' + self.fy] \
                   + row['IPNGTD' + self.fy] + row['HHTOTD' + self.fy]

        df['TOTEXP' + self.fy] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP' + self.fy] < 10.0
        df.loc[lessE, 'TOTEXP' + self.fy] = 0.0
        moreE = df['TOTEXP' + self.fy] >= 10.0
        df.loc[moreE, 'TOTEXP' + self.fy] = 1.0

        df = df.rename(columns={'TOTEXP' + self.fy: 'UTILIZATION'})

        return df


class MEPSDataset19(MEPSDataset):
    """ panel 19 fy 2015 """

    def __init__(self):
        meta = json.load(open("data/meps/meta19.json"))
        df = pd.read_csv(meta["train_path"], sep=",")
        super(MEPSDataset19, self).__init__(panel=19, fy="15", df=df, **meta, shuffle=False)


class MEPSDataset20(MEPSDataset):
    """ panel 20 fy 2015 """

    def __init__(self):
        meta = json.load(open("data/meps/meta20.json"))
        df = pd.read_csv(meta["train_path"], sep=",")
        super(MEPSDataset20, self).__init__(panel=20, fy="15", df=df, **meta, shuffle=False)


class MEPSDataset21(MEPSDataset):
    """ panel 21 fy 2016 """

    def __init__(self):
        meta = json.load(open("data/meps/meta21.json"))
        df = pd.read_csv(meta["train_path"], sep=",")
        super(MEPSDataset21, self).__init__(panel=21, fy="16", df=df, **meta, shuffle=False)

'''
class Credit(Dataset):
    """ c """

    def __init__(self):
        meta = json.load(open("data/credit/meta.json"))
        meta["categorical_feat"] = meta["categorical_feat"].split(",")
        df = pd.read_excel(meta["train_path"], header=1, index_col=0)
        df["SEX"] = df["SEX"].astype("str")
        super(Credit, self).__init__(name="Credit", df=df, **meta, shuffle=False)
'''


def fair_stat(data: DataTemplate2):
    s_cnt = Counter(data.s_train)
    s_pos_cnt = {s: 0. for s in s_cnt.keys()}
    for i in range(data.num_train):
        if data.y_train[i] == 1:
            s_pos_cnt[data.s_train[i]] += 1

    print("-" * 10, "Statistic of fairness")
    for s in s_cnt.keys():
        print("Grp. %d - #instance: %d; #pos.: %d; ratio: %.3f" % (s, s_cnt[s], s_pos_cnt[s], s_pos_cnt[s] / s_cnt[s]))

    print("Overall - #instance: %d; #pos.: %d; ratio: %.3f" % (sum(s_cnt.values()), sum(s_pos_cnt.values()),
                                                               sum(s_pos_cnt.values()) / sum(s_cnt.values())))

    return


def fetch_data2(name):
    if name == "adult":
        return AdultDataset().data
    elif name == "comm":
        return CommDataset().data
    elif name == "compas":
        return CompasDataset().data
    elif name == "bank":
        return Bank().data
    elif name == "german":
        return GermanDataset().data
    elif name == "meps19":
        return MEPSDataset19().data
    elif name == "meps20":
        return MEPSDataset20().data
    elif name == "meps21":
        return MEPSDataset21().data
    elif name == "credit":
        return Credit().data
    elif name == "dist_shift_adult":
        return DistShiftAdult().data
    elif name == "dist_shift_adult_time":
        return DistShiftAdultTime().data
    elif name == "dist_shift_adult_loc":
        return DistShiftAdultLoc().data
    elif name == "toy":
        return Toy().data
    elif name == "synthetic":
        return Synthetic().data
    elif name == "acs":
        return ACSDataset().data
    elif name == "celeba":
        return CelebA().data
    elif name == "nlp":
        return NLP().data
    elif name == "raa_attack_compas":
        return RAA_Attack_Compas().data
    elif name == "nraa_attack_compas":
        return NRAA_Attack_Compas().data
    elif name == "iaf_attack_compas":
        return IAF_Attack_Compas().data
    elif name == "solans_attack_compas":
        return Solans_Attack_Compas().data
    elif name == "raa_attack_drug":
        return RAA_Attack_Drug().data
    elif name == "nraa_attack_drug":
        return NRAA_Attack_Drug().data
    elif name == "iaf_attack_drug":
        return IAF_Attack_Drug().data
    elif name == "solans_attack_drug":
        return Solans_Attack_Drug().data
    elif name == "raa_attack_german":
        return RAA_Attack_German().data
    elif name == "nraa_attack_german":
        return NRAA_Attack_German().data
    else:
        raise ValueError


if __name__ == "__main__":
    data = fetch_data2("german")
    fair_stat(data)
