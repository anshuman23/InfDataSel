from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X_tr_c, Y, S = np.load('nraa_attack_data/X_train_clean.npy'), np.load('nraa_attack_data/y_train_clean.npy'), np.load('nraa_attack_data/s_train_clean.npy')
print(X_tr_c.shape,Y.shape,S.shape)

y_tr_c = []
for i,el in enumerate(Y):
    if el == 0:
        y_tr_c.append("N")
    elif el == 1:
        y_tr_c.append("Y")
    else:
        print(el)
y_tr_c = np.array(y_tr_c).reshape((-1,1))

s_tr_c = []
for i,el in enumerate(S):
    if el == 1:
        s_tr_c.append("A")
    elif el == 0:
        s_tr_c.append("D")
s_tr_c = np.array(s_tr_c).reshape((-1,1))

cols = [x+1 for x in range(X_tr_c.shape[1])]

X_tr_c = np.append(X_tr_c, s_tr_c, 1)
X_tr_c = np.append(X_tr_c, y_tr_c, 1)

cols.append("advantage")
cols.append("outcome")
X_tr = pd.DataFrame(X_tr_c, columns=cols)



X_te_c, Y, S = np.load('nraa_attack_data/X_test.npy'), np.load('nraa_attack_data/y_test.npy'), np.load('nraa_attack_data/s_test.npy')
print(X_te_c.shape,Y.shape,S.shape)

y_te_c = []
for i,el in enumerate(Y):
    if el == 0:
        y_te_c.append("N")
    elif el == 1:
        y_te_c.append("Y")
y_te_c = np.array(y_te_c).reshape((-1,1))

s_te_c = []
for i,el in enumerate(S):
    if el == 1:
        s_te_c.append("A")
    elif el == 0:
        s_te_c.append("D")
s_te_c = np.array(s_te_c).reshape((-1,1))

cols = [x+1 for x in range(X_te_c.shape[1])]

X_te_c = np.append(X_te_c, s_te_c, 1)
X_te_c = np.append(X_te_c, y_te_c, 1)

cols.append("advantage")
cols.append("outcome")
X_te = pd.DataFrame(X_te_c, columns=cols)


print(X_tr)

print(X_te)

X_tr.to_csv('train_clean.csv', index=False)
X_te.to_csv('test.csv', index=False)
