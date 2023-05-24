from folktables import ACSDataSource, ACSIncome
import folktables
import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ACS (Adult/Census) Data Generator')
    parser.add_argument('--state', type=str, default="CA", help="source state")
    parser.add_argument('--year', type=str, default="2015", help="survey year")
    parser.add_argument('--dataset', type=str, default="error", help="name of the dataset")
    parser.add_argument('--type', type=str, default="train", help="train/test")
    parser.add_argument('--num_samples', type=int, default=12000, help="num samples")

    args = parser.parse_args()

    return args

args = parse_args()

#FOR DIST_SHIFT_ADULT ==> train -> CA, 2014 (18k); test -> MI, 2018 (12k)
#FOR DIST_SHIFT_ADULT_LOC ==> train -> CA, 2014 (18k); test -> MI, 2014 (12k)
#FOR DIST_SHIFT_ADULT_TIME ==> train -> CA, 2014 (18k); test -> CA, 2018 (12k)
data_source = ACSDataSource(survey_year=args.year, horizon='1-Year', survey='person') 
ca_data = data_source.get_data(states=[args.state], download=True)


ACSIncome = folktables.BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,    
    group='SEX',
    preprocess=folktables.adult_filter
)

ca_features, ca_labels, ca_groups = ACSIncome.df_to_numpy(ca_data)

ca_groups = ca_groups - 1
ca_groups = 1 - ca_groups

new_ca_groups = []
for ca_group in ca_groups:
    if ca_group == 1:
        new_ca_groups.append("Male")
    else:
        new_ca_groups.append("Female")

new_ca_labels = []
for ca_label in ca_labels:
    if ca_label == True:
        new_ca_labels.append("Y")
    else:
        new_ca_labels.append("N")


df = pd.DataFrame(data = ca_features, columns = ["AGEP","COW","SCHL","MAR","OCCP","POBP","RELP","WKHP","SEX","RAC1P"])
df = df.dropna().reset_index(drop=True)
df["SEX"] = new_ca_groups
df["PINCP"] = new_ca_labels
df = df.sample(args.num_samples)
print(df)

df.to_csv("data/"+args.dataset+"/"+args.type+".csv", index=False, header=False)
