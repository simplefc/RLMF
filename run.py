import multiprocessing
import os
import sys
import time
import numpy as np

sys.path.append('src')
# Build external model
if not os.path.isfile('src/core.pyd'):
    print('Lack of core.pyd (built from the C++ module).')
    print('Please first build the C++ code into core.pyd by using: ')
    print('>> python setup.py build_ext --inplace')
    sys.exit()

import evaluator
import dataloader
from utilities import logger, initConfig

#########################################################
# config area
#
para = {'dataType': 'rt',
        'dataPath': './data/',
        'outPath': 'result/',
        'metrics': ['MAE', 'RMSE'],  # delete where appropriate
        'density': list(np.arange(0.05, 0.31, 0.05)),  # matrix density
        'rounds': 20,  # how many runs are performed at each matrix density
        'dimension': 10,  # dimension of the latent factors
        'gamma': 1,
        'alphaInit': 0.003,  # initial learning rate
        'decayRate': 0.9,
        'decaySteps': 50,
        'lambda': 5,  # regularization parameter
        'eta': 5,  # regularization parameter
        'n_clusters': 4,
        'maxIter': 1000,  # the max iterations
        'saveTimeInfo': False,  # whether to keep track of the running time
        'saveLog': True,  # whether to save log into file
        'debugMode': False,  # whether to record the debug info
        'parallelMode': False  # whether to leverage multiprocessing for speedup
        }

initConfig(para)


#########################################################

def main():
    startTime = time.time()  # start timing
    logger.info('==============================================')
    logger.info('RLMF: Reputation and Location Aware Matrix Factorization.')

    # load the dataset
    dataMatrix = dataloader.load(para)
    user_country_matrix = np.load(para['dataPath'] + 'user_country_matrix.npy')
    I_outlier = np.load(para['dataPath'] + 'Full_I_outlier.npy')
    logger.info('Loading data done.')
    # run for each density
    if para['parallelMode']:  # run on multiple processes
        pool = multiprocessing.Pool()
        for density in para['density']:
            pool.apply_async(evaluator.execute, (dataMatrix, I_outlier, user_country_matrix, density, para))
        pool.close()
        pool.join()
    else:  # run on single processes
        for density in para['density']:
            evaluator.execute(dataMatrix, I_outlier, user_country_matrix, density, para)
    logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
                              time.gmtime(time.time() - startTime)))
    logger.info('==============================================')
    sys.path.remove('src')


if __name__ == '__main__':
    main()
