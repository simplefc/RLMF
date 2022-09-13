import random
import warnings
import time
from numpy import linalg as LA
from sklearn.cluster import KMeans

import core
from utilities import *


########################################################
# Function to run the prediction approach at each density
# 
def execute(matrix, I_outlier, user_country_matrix, density, para):
    start_time = time.time()
    num_user = matrix.shape[0]
    num_service = matrix.shape[1]
    rounds = para['rounds']
    n_clusters = para['n_clusters']
    debug_mode = para['debugMode']

    logger.info('Data matrix size: %d users * %d services' % (num_user, num_service))
    logger.info('Run the algorithm for %d rounds: matrix density = %.2f.' % (rounds, density))
    eval_results = np.zeros((rounds, len(para['metrics'])))
    time_results = np.zeros((rounds, 1))
    for k in range(rounds):
        logger.info('----------------------------------------------')
        logger.info('%d-round starts.' % (k + 1))
        logger.info('----------------------------------------------')

        # remove the entries of data matrix to generate train_matrix and test_matrix
        # use k as random seed
        (train_matrix, test_matrix) = removeEntries(matrix, density, k)
        test_matrix = merge_test_outlier(I_outlier, test_matrix)
        logger.info('Removing data entries done.')
        # invocation to the prediction function
        iter_start_time = time.time()  # to record the running time for one round
        weight_matrix = get_user_reputation_weight_matrix(user_country_matrix, train_matrix, n_clusters)
        loss, err = core.predict(train_matrix, test_matrix, weight_matrix, para)
        if debug_mode:
            curve(loss, str(k) + "round")

        mae = err[:, 0]
        rmse = err[:, 1]
        rmse = rmse[rmse > 0]
        min_rmse = np.min(rmse)
        idx = np.where(rmse == min_rmse)
        result = err[idx]
        eval_results[k, :] = result
        time_results[k] = time.time() - iter_start_time
        logger.info('%d-round done. Running time: %.2f sec' % (k + 1, time_results[k]))
        logger.info('----------------------------------------------')

    out_file = '%s%sResult_%.2f.txt' % (para['outPath'], para['dataType'], density)
    saveResult(out_file, eval_results, time_results, para)
    logger.info('Config density = %.2f done. Running time: %.2f sec'
                % (density, time.time() - start_time))
    logger.info('==============================================')


########################################################

########################################################
def get_user_reputation_weight_matrix(user_country_matrix, train_matrix, n_clusters):
    num_service = train_matrix.shape[1]
    num_user = train_matrix.shape[0]
    reputation = np.zeros((num_user, 2), dtype=int)
    for sid in range(0, num_service):
        # cluster all users for service j
        try:
            data = train_matrix[:, sid]
            real_idx = np.where(data > 0)
            real_data = data[real_idx].reshape(-1, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters).fit(real_data)
            labels = km.labels_
            _label = np.argmax(np.bincount(labels))
            # maximum cluster
            max_cluster = real_data[labels == _label]
            mean = max_cluster.mean()
            std = max_cluster.std()
            for uid in real_idx[0]:
                # positive feedback
                if abs(train_matrix[uid][sid] - mean) <= 3 * std:
                    reputation[uid][0] += 1
                # negative feedback
                else:
                    reputation[uid][1] += 1
        except Exception as e:
            # fail to cluster because of too little QoS data
            pass
    # calculate reputation
    user_reputation = (reputation[:, 0] + 1) / (reputation[:, 0] + reputation[:, 1] + 2)
    weight_matrix = user_country_matrix * user_reputation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight_matrix = weight_matrix / weight_matrix.sum(1).reshape(1, -1)
    weight_matrix[np.isnan(weight_matrix)] = 0
    return weight_matrix


########################################################

def merge_test_outlier(I_outlier, test_matrix):
    merge_matrix = test_matrix.copy()
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            if test_matrix[i][j] != 0 and I_outlier[i][j] == 1:
                merge_matrix[i][j] = 0
    return merge_matrix


########################################################
# Function to remove the entries of data matrix
# Return the trainMatrix and the corresponding testing data
#
def removeEntries(matrix, density, seedID):
    (vecX, vecY) = np.where(matrix > 0)
    vecXY = np.c_[vecX, vecY]
    numRecords = vecX.size
    numAll = matrix.size
    random.seed(seedID)
    randomSequence = list(range(0, numRecords))
    random.shuffle(randomSequence)  # one random sequence per round
    numTrain = int(numAll * density)
    # by default, we set the remaining QoS records as testing data
    numTest = numRecords - numTrain
    trainXY = vecXY[randomSequence[0: numTrain], :]
    testXY = vecXY[randomSequence[- numTest:], :]

    trainMatrix = np.zeros(matrix.shape)
    trainMatrix[trainXY[:, 0], trainXY[:, 1]] = matrix[trainXY[:, 0], trainXY[:, 1]]
    testMatrix = np.zeros(matrix.shape)
    testMatrix[testXY[:, 0], testXY[:, 1]] = matrix[testXY[:, 0], testXY[:, 1]]

    # ignore invalid testing data
    idxX = (np.sum(trainMatrix, axis=1) == 0)
    testMatrix[idxX, :] = 0
    idxY = (np.sum(trainMatrix, axis=0) == 0)
    testMatrix[:, idxY] = 0
    return trainMatrix, testMatrix


########################################################


########################################################
# Function to compute the evaluation metrics
# Return an array of metric values
#
def errMetric(testMatrix, predMatrix, metrics):
    result = []
    (testVecX, testVecY) = np.where(testMatrix)
    testVec = testMatrix[testVecX, testVecY]
    predVec = predMatrix[testVecX, testVecY]
    absError = np.absolute(predVec - testVec)
    mae = np.average(absError)
    for metric in metrics:
        if 'MAE' == metric:
            result = np.append(result, mae)
        elif 'RMSE' == metric:
            rmse = LA.norm(absError) / np.sqrt(absError.size)
            result = np.append(result, rmse)
    return result
########################################################
