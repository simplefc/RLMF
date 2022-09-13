# RLMF: Reputation and Location Aware Matrix Factorization

This repo maintains the implementation of the RLMF approach for Web service QoS prediction.

## Usage

1. Dataset

   The adopted dataset can be downloaded from [WSDREAM](https://github.com/wsdream/wsdream-dataset).


2. Preparation

   ```
   python data.py
   ```

3. Train the model

   * RLMF is implemented in Cython, so we need to compile the code at first

     ```
     python setup.py build_ext --inplace
     ```

   * Then, train the model

     ```
     python run.py
     ```

     The configurations can be changed in run.py.

### Requirements

 + Python 3.8

 + Cython 0.29.28
