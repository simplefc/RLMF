import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

## global
logger = logging.getLogger('logger')


########################################################
# Config the working paths and set up logger
#
def initConfig(para):
    config = {'exeFile': os.path.basename(sys.argv[0]),
              'workPath': os.path.abspath('.'),
              'srcPath': os.path.abspath('src/'),
              'dataPath': os.path.abspath('../data/'),
              'logFile': os.path.basename(sys.argv[0]) + '.log'}

    # add result folder
    if not os.path.exists(para['outPath']):
        os.makedirs(para['outPath'])

    ## set up logger to record runtime info
    if para['debugMode']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # log to console
    cmdhandler = logging.StreamHandler()
    cmdhandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s (pid-%(process)d): %(message)s')
    cmdhandler.setFormatter(formatter)
    logger.addHandler(cmdhandler)
    # log to file
    if para['saveLog']:
        filehandler = logging.FileHandler(config['logFile'])
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    logger.info('==========================================')
    logger.info('Config:')
    config.update(para)
    for name in config:
        logger.info('%s = %s' % (name, config[name]))


########################################################


########################################################
# Save the evaluation results into file
#
def saveResult(outfile, result, timeinfo, para):
    fileID = open(outfile, 'w')
    fileID.write('Metric: ')
    for metric in para['metrics']:
        if isinstance(metric, str):
            fileID.write('| %s\t' % metric)
        elif isinstance(metric, tuple):
            if 'NDCG' == metric[0]:
                for topK in metric[1]:
                    fileID.write('| NDCG%s\t' % topK)
    avgResult = np.average(result, axis=0)
    fileID.write('\nAvg:\t')
    np.savetxt(fileID, np.matrix(avgResult), fmt='%.4f', delimiter='\t')
    stdResult = np.std(result, axis=0)
    fileID.write('Std:\t')
    np.savetxt(fileID, np.matrix(stdResult), fmt='%.4f', delimiter='\t')
    fileID.write('\n==========================================\n')
    fileID.write('Detailed results for %d rounds:\n' % result.shape[0])
    np.savetxt(fileID, result, fmt='%.4f', delimiter='\t')
    fileID.close()

    if para['saveTimeInfo']:
        fileID = open(outfile + '_time.txt', 'w')
        fileID.write('Running time:\nAvg:\t%.4f\n' % np.average(timeinfo))
        fileID.write('Std:\t%.4f\n' % np.std(timeinfo))
        fileID.write('\n==========================================\n')
        fileID.write('Detailed results for %d rounds:\n' % timeinfo.shape[0])
        np.savetxt(fileID, np.matrix(timeinfo), fmt='%.4f')
        fileID.close()


########################################################

def curve(costs, title="Convergence curve"):
    """
    show the cost value trend
    :param costs: cost value list
    """
    x = range(len(costs))
    plt.plot(x, costs, color='r', linewidth=3)
    plt.title(title)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
