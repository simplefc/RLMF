#include <iostream>
#include <cstring>
#include <cmath>
#include "RLMF.h"
#include <vector>
#include <unordered_map>

using namespace std;

/**
 *
 * @param removedData training set
 * @param numUser
 * @param numService
 * @param dim: latent vector dimension
 * @param lmda: regularization term parameter
 * @param eta: user regularization term parameter
 * @param maxIter: maximum iteration number
 * @param alphaInit: learning rate
 * @param Udata: user latent matrix
 * @param Sdata: service latent matrix
 */
void RLMF(double *removedData, double *testData, double *weightData, int numUser, int numService, int dim, double gamma,
          double lmda, double eta, int maxIter, double alphaInit, double decayRate, int decaySteps, double *Udata,
          double *Sdata, double *lossData, double *errData, bool debugMode) {
    // --- transfer the 1D pointer to 2D array pointer
    double **removedMatrix = vector2Matrix(removedData, numUser, numService);
    double **testMatrix = vector2Matrix(testData, numUser, numService);
    double **weightMatrix = vector2Matrix(weightData, numUser, numUser);
    double **U = vector2Matrix(Udata, numUser, dim);
    double **S = vector2Matrix(Sdata, numService, dim);
    double **errResult = vector2Matrix(errData, maxIter, 2);

    // --- create a set of temporal matries
    double **gradU = createMatrix(numUser, dim);
    double **gradS = createMatrix(numService, dim);
    unordered_map<int, vector<int>> userNeighbors = findNeighbors(weightMatrix, numUser);

    // --- iterate by standard gradient descent algorithm
    int iter, i, j, k;
    double alpha;
    for (iter = 0; iter < maxIter; iter++) {
        alpha = alphaInit * pow(decayRate, iter / decaySteps);

        // update gradients
        gradLoss(U, S, removedMatrix, weightMatrix, userNeighbors, gradU, gradS, gamma, lmda, eta, numUser, numService,
                 dim);

        // gradient descent updates
        for (k = 0; k < dim; k++) {
            // update U
            for (i = 0; i < numUser; i++) {
                U[i][k] -= alpha * gradU[i][k];
            }
            // update S
            for (j = 0; j < numService; j++) {
                S[j][k] -= alpha * gradS[j][k];
            }
        }

        if (debugMode) {
            // update loss value
            lossData[iter] = loss(U, S, removedMatrix, weightMatrix, userNeighbors, gamma, lmda, eta, numUser,
                                  numService, dim);
        }

        errMetric(testMatrix, U, S, numUser, numService, dim, errResult, iter);
    }

    delete2DMatrix(gradU);
    delete2DMatrix(gradS);
    delete ((char *) U);
    delete ((char *) S);
    delete ((char *) removedMatrix);
    delete ((char *) testMatrix);
    delete ((char *) weightMatrix);
    delete ((char *) errResult);
}

double square(double x) {
    return x * x;
}

unordered_map<int, vector<int>> findNeighbors(double **weightMatrix, int numUser) {
    unordered_map<int, vector<int>> user_neighbors;
    for (int i = 0; i < numUser; i++) {
        vector<int> neighbors;
        for (int j = 0; j < numUser; j++) {
            if (weightMatrix[i][j] != 0) {
                neighbors.push_back(j);
            }
        }
        user_neighbors[i] = neighbors;
    }
    return user_neighbors;
}

double loss(double **U, double **S, double **removedMatrix, double **weightMatrix,
            unordered_map<int, vector<int>> &userNeighbors, double gamma,
            double lmda, double eta, int numUser, int numService, int dim) {
    int i, j, k;
    double loss = 0;
    double **predMatrix = createMatrix(numUser, numService);

    // update predMatrix
    U_dot_S(removedMatrix, U, S, numUser, numService, dim, predMatrix);
    double gamma2 = gamma * gamma;
    double res;
    // cost
    for (i = 0; i < numUser; i++) {
        for (j = 0; j < numService; j++) {
            if (removedMatrix[i][j] != 0) {
                res = removedMatrix[i][j] - predMatrix[i][j];
                loss += 0.5 * log((gamma2 + square(res)) / gamma2);
            }
        }
    }

    // L2 regularization
    for (k = 0; k < dim; k++) {
        for (i = 0; i < numUser; i++) {
            loss += 0.5 * lmda * square(U[i][k]);
        }
        for (j = 0; j < numService; j++) {
            loss += 0.5 * lmda * square(S[j][k]);
        }
    }

    double neighbor_k;
    for (i = 0; i < numUser; i++) {
        vector<int> neighbors = userNeighbors[i];
        for (k = 0; k < dim; k++) {
            neighbor_k = 0;
            for (int ne: neighbors) {
                neighbor_k += U[ne][k] * weightMatrix[i][ne];
            }
            loss += 0.5 * eta * square(U[i][k] - neighbor_k);
        }
    }

    delete2DMatrix(predMatrix);
    return loss;
}


/**
 * calculate gradient: gradU and gradS
 * @param U
 * @param S
 * @param removedMatrix
 * @param gradU
 * @param gradS
 * @param lmda
 * @param numUser
 * @param numService
 * @param dim
 */
void gradLoss(double **U, double **S, double **removedMatrix, double **weightMatrix,
              unordered_map<int, vector<int>> &userNeighbors, double **gradU,
              double **gradS, double gamma, double lmda, double eta, int numUser, int numService, int dim) {
    int i, j, k;
    double grad;
    double gamma2 = gamma * gamma;
    double res;
    double neighbor_k;

    // gradU
    for (i = 0; i < numUser; i++) {
        vector<int> neighbors = userNeighbors[i];
        for (k = 0; k < dim; k++) {
            grad = 0;
            for (j = 0; j < numService; j++) {
                if (removedMatrix[i][j] != 0) {
                    res = removedMatrix[i][j] - dotProduct(U[i], S[j], dim);
                    grad += (res / (gamma2 + square(res))) * (-S[j][k]);
                }
            }
            grad += lmda * U[i][k];
            neighbor_k = 0;
            for (int ne: neighbors) {
                neighbor_k += U[ne][k] * weightMatrix[i][ne];
            }
            grad += eta * (U[i][k] - neighbor_k);
            gradU[i][k] = grad;
        }
    }

    // gradS
    for (j = 0; j < numService; j++) {
        for (k = 0; k < dim; k++) {
            grad = 0;
            for (i = 0; i < numUser; i++) {
                if (removedMatrix[i][j] != 0) {
                    res = removedMatrix[i][j] - dotProduct(U[i], S[j], dim);
                    grad += (res / (gamma2 + square(res))) * (-U[i][k]);
                }
            }
            grad += lmda * S[j][k];
            gradS[j][k] = grad;
        }
    }

}

/**
 * matrix multiply
 * @param removedMatrix
 * @param U
 * @param S
 * @param numUser
 * @param numService
 * @param dim
 * @param predMatrix
 */
void U_dot_S(double **removedMatrix, double **U, double **S, int numUser,
             int numService, int dim, double **predMatrix) {
    int i, j;
    for (i = 0; i < numUser; i++) {
        for (j = 0; j < numService; j++) {
            if (removedMatrix[i][j] != 0) {
                predMatrix[i][j] = dotProduct(U[i], S[j], dim);
            }
        }
    }
}

void errMetric(double **testMatrix, double **U, double **S, int numUser,
               int numService, int dim, double **errResult, int iter) {
    int i, j;
    int N = 0;
    double absError = 0;
    double squareError = 0;
    double pred;
    for (i = 0; i < numUser; i++) {
        for (j = 0; j < numService; j++) {
            if (testMatrix[i][j] != 0) {
                N++;
                pred = dotProduct(U[i], S[j], dim);
                absError += abs(testMatrix[i][j] - pred);
                squareError += square(testMatrix[i][j] - pred);
            }
        }
    }
    errResult[iter][0] = absError / N;
    errResult[iter][1] = sqrt(squareError / N);
}

/**
 * vector to matrix
 * @param vector
 * @param row
 * @param col
 * @return
 */
double **vector2Matrix(double *vector, int row, int col) {
    double **matrix = new double *[row];

    int i;
    for (i = 0; i < row; i++) {
        matrix[i] = vector + i * col;
    }
    return matrix;
}

/**
 * vector dot product
 * @param vec1
 * @param vec2
 * @param len
 * @return
 */
double dotProduct(double *vec1, double *vec2, int len) {
    double product = 0;
    int i;
    for (i = 0; i < len; i++) {
        product += vec1[i] * vec2[i];
    }
    return product;
}

/**
 * create matrix
 * @param row
 * @param col
 * @return
 */
double **createMatrix(int row, int col) {
    double **matrix = new double *[row];
    matrix[0] = new double[row * col];
    memset(matrix[0], 0, row * col * sizeof(double)); // Initialization
    int i;
    for (i = 1; i < row; i++) {
        matrix[i] = matrix[i - 1] + col;
    }
    return matrix;
}


void delete2DMatrix(double **ptr) {
    delete ptr[0];
    delete ptr;
}




