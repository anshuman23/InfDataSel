from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X, Y, S = np.load('raw_data/bank/X.npy'), np.load('raw_data/bank/y.npy'), np.load('raw_data/bank/s.npy')
print(X.shape,Y.shape,S.shape)

y = []
for i,el in enumerate(Y):
    if el == 0:
        y.append("N")
    elif el == 1:
        y.append("Y")
y = np.array(y).reshape((-1,1))

s = []
for i,el in enumerate(S):
    if el == 1:
        s.append("F")
    elif el == 0:
        s.append("M")
s = np.array(s).reshape((-1,1))

cols = [x+1 for x in range(X.shape[1])]

X = np.append(X, s, 1)
X = np.append(X, y, 1)

cols.append("sex")
cols.append("outcome")
X_df = pd.DataFrame(X, columns=cols)
#print(X_df)

X_tr, X_te = train_test_split(X_df, random_state=42, test_size=0.4)

print(X_tr)

print(X_te)

X_tr.to_csv('train.csv', index=False)
X_te.to_csv('test.csv', index=False)
