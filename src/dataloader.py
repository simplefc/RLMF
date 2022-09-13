from utilities import *


########################################################
# Function to load the dataset
#
def load(para):
    datafile = para['dataPath'] + para['dataType'] + 'Matrix.txt'
    logger.info('Load data: %s' % datafile)
    dataMatrix = np.loadtxt(datafile)
    dataMatrix = preprocess(dataMatrix, para)
    return dataMatrix


########################################################


########################################################
# Function to preprocess the dataset
# delete the invalid values
# 
def preprocess(matrix, para):
    if para['dataType'] == 'rt':
        matrix = np.where(matrix == 0, -1, matrix)
    elif para['dataType'] == 'tp':
        matrix = np.where(matrix == 0, -1, matrix)
    return matrix
########################################################
