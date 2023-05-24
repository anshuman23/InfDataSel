from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X_tr_p, Y, S = np.load('nraa_attack_data/X_train_adv.npy'), np.load('nraa_attack_data/y_train_adv.npy'), np.load('nraa_attack_data/s_train_adv.npy')
print(X_tr_p.shape,Y.shape,S.shape)

y_tr_p = []
for i,el in enumerate(Y):
    if el == 0:
        y_tr_p.append("N")
    elif el == 1:
        y_tr_p.append("Y")
y_tr_p = np.array(y_tr_p).reshape((-1,1))

s_tr_p = []
for i,el in enumerate(S):
    if el == 1:
        s_tr_p.append("A")
    elif el == 0:
        s_tr_p.append("D")
s_tr_p = np.array(s_tr_p).reshape((-1,1))

cols = [x+1 for x in range(X_tr_p.shape[1])]

X_tr_p = np.append(X_tr_p, s_tr_p, 1)
X_tr_p = np.append(X_tr_p, y_tr_p, 1)

cols.append("advantage")
cols.append("outcome")
X_tr = pd.DataFrame(X_tr_p, columns=cols)

print(X_tr)

X_tr.to_csv('train_poisoned.csv', index=False)
