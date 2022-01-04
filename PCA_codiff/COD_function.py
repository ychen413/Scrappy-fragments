# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def operator_codiff(a, b, otype='minimum'):
    # Input: a, b are pure values or vectors (m, ) | (1, m) | (m, 1)
    # Output: (m, ) | (1, m) | (m, 1) with [operator_codiff(a[i], b[i])], i=1, ..., m
    
    if otype == 'minimum':
        sol = np.sign(a*b) * np.minimum(np.abs(a), np.abs(b))
    elif otype == 'additive':
        sol = np.sign(a*b) * (np.abs(a) + np.abs(b))
    return sol

def operator_codiff_vector(va, vb, otype='minimum'):
    # Input: va, vb are vectors (m, ) | (1, m) | (m, 1)
    # Output: pure value with sum(operator_codiff(a[i], b[i])), i=1, ..., m
    
    if otype == 'minimum':
        sol = np.sum(np.sign(va*vb) * np.minimum(np.abs(va), np.abs(vb)))
    elif otype == 'additive':
        sol = np.sum(np.sign(va*vb) * (np.abs(va) + np.abs(vb)))
    return sol

def matrix_codiff(a, cod_type='minimum'):
    # Input: a is a matrix (m, n) with #m data (la0) & #n features (la1)
    # Output: (n, n) codifference matrix 
    
    la0 = np.size(a, axis=0)
    la1 = np.size(a, axis=1)
    ma = np.empty((la1, la1))
    for l0 in np.arange(la1):
        for l1 in np.arange(la1):
            ma[l0, l1] = operator_codiff_vector(a[:, l0], a[:, l1], cod_type)    
    
    return ma / (la0-1.0)

def pca_cod(dataset, dim_after, cod_type='minimum'):
    # Input: dataset (m, n) with [#m data, #n features]
    # Output: #dim_after principle vectors
    #         the corresponding low-dimensional data before moving back from mean
    # By codifference matrix
    
    p_mean = np.mean(dataset, axis=0)
    d_norm = dataset - p_mean
    m_cod = matrix_codiff(d_norm, cod_type)
    e_val, e_vec = np.linalg.eig(m_cod)

    pc = e_vec[:, np.argsort(e_val)[::-1][:dim_after]]
    d_lowdim = d_norm.dot(pc)
    
    return pc.T, d_lowdim, p_mean

def pca_cov(dataset, dim_after):
    # Input: dataset (m, n) with [#m data, #n features]
    # Output: #dim_after principle vectors
    #         the corresponding low-dimensional data before moving back from mean
    # By coveriance matrix
    
    p_mean = np.mean(dataset, axis=0)
    d_norm = dataset - p_mean
#    d_norm = d_norm / np.std(d_norm)
    m_cov = np.cov(d_norm.T)
    e_val, e_vec = np.linalg.eig(m_cov)
    
    eig_explained_variance(e_val)
    
    pc = e_vec[:, np.argsort(e_val)[::-1][:dim_after]]
    d_lowdim = d_norm.dot(pc)
    
    return pc.T, d_lowdim, p_mean

def eig_explained_variance(eigenvalues):
    eig = np.sort(eigenvalues)[::-1]
    total = np.sum(eig)
    exp_var = [line / total * 100 for line in eig] 
    exp_var_cum = np.cumsum(exp_var)
    ind = np.arange(np.size(exp_var))+1
    
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
    
        plt.bar(ind, exp_var, alpha=0.5, align='center', \
                label='individual explained variance')
        plt.step(ind, exp_var_cum, where='mid', \
                 label='cumulative explained variance')
        plt.xticks(ind)
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
        plt.tight_layout()
        
    pass
