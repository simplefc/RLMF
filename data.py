import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import IsolationForest


def get_user_country_matrix():
    userlist = pd.read_csv("data/userlist.txt", sep="\t")
    num_user = userlist.shape[0]
    matrix = np.zeros((num_user, num_user), dtype=int)

    country_user = userlist.groupby('[Country]')['[User ID]'].unique()
    for key in country_user.keys():
        arr = country_user[key]
        pairs = list(itertools.permutations(arr, 2))
        for u, v in pairs:
            matrix[u][v] = 1
    np.save("data/user_country_matrix.npy", matrix)


def get_outlier_matrix(outliers_fraction):
    '''
    output: Indicator Matrix (0: normal; 1: outlier)
    '''
    R = load_data("data/rtMatrix.txt")
    m, n = R.shape
    I_outlier = np.zeros([m, n])
    rng = np.random.RandomState(42)
    x = []
    x_ind = []
    for i in range(m):
        for j in range(n):
            if R[i][j] >= 0:
                x.append(R[i][j])
                x_ind.append(i * n + j)

    x = np.array(x)
    x = x.reshape(-1, 1)

    clf = IsolationForest(max_samples=len(x), random_state=rng, contamination=outliers_fraction)
    clf.fit(x)
    y_pred_train = clf.predict(x)
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            row = int(x_ind[i] / n)
            col = int(x_ind[i] % n)
            I_outlier[row][col] = 1

    filename = "data/Full_I_outlier.npy"
    np.save(filename, I_outlier)
    print("============outliers_fraction %s DONE ==============" % outliers_fraction)


def load_data(filedir):
    R = []
    with open(filedir) as fin:
        for line in fin:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R


if __name__ == '__main__':
    outliers_fraction = 0.1
    get_outlier_matrix(outliers_fraction)

    get_user_country_matrix()
