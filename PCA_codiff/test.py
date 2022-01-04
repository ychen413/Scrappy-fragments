# -*- coding: utf-8 -*-
"""
Created on Thu May 23 01:29:00 2019

@author: deep362
"""

import numpy as np
import matplotlib.pyplot as plt
import COD_function as cdf
import pandas as pd

def plot_line(vec, p_mean, xrange, set_line='r--', label_='PCA'):
    sl = vec[:, 1] / vec[:, 0]
    y =  sl * xrange + (p_mean[1] - sl * p_mean[0])
#    line = np.concatenate((x, y))
    plt.plot(xrange, y, set_line, label=label_)
    pass

def plot_xyaxis(xrange, yrange):
    plt.plot(xrange, [0, 0], 'k-')
    plt.plot([0, 0], yrange, 'k-')
    pass

# ===========================
# Initial test
#a = np.array([[-1, -3],[-1, 0], [0, 0], [0, 1], [3, 1]])
#
#pc_cod, d_cod, mean_cod = cdf.pca_cod(a, 1, cod_type='additive')
#pc_cov, d_cov, mean_cov = cdf.pca_cov(a, 1)
#
#d_cod_ = d_cod + mean_cod
#d_cov_ = d_cov + mean_cov
#
#plt.figure(figsize=(6, 6))
#xr = np.array([-5, 5])
#yr = np.array([-5, 5])
#
#plt.plot(a[:,0],a[:,1], 'bo')
#plot_xyaxis(xr, yr)
#plot_line(pc_cod, mean_cod, xr)
#plt.plot(d_cod_[:, 0], d_cod_[:, 1], 'bx')
#
#plt.figure(figsize=(6, 6))
#xr = np.array([-5, 5])
#yr = np.array([-5, 5])
#
#plt.plot(a[:,0],a[:,1], 'bo')
#plot_xyaxis(xr, yr)
#plot_line(pc_cov, mean_cov, xr)
#plt.plot(d_cov_[:, 0], d_cov_[:, 1], 'bx')

# ============================
# Iris data
#loc_iris = 'D:/E/Research_USA/dataset/dataset_Iris/iris.xlsx'
#sheet_name = 'iris'
#df = pd.read_excel(loc_iris, sheetname=sheet_name, header=None)
#dataset = df.values  
#
#mask0 = dataset[:, -1]=='Iris-setosa'
#mask1 = dataset[:, -1]=='Iris-versicolor'
#mask2 = dataset[:, -1]=='Iris-virginica'
#
#dataset[mask0, -1] = 0
#dataset[mask1, -1] = 1
#dataset[mask2, -1] = 2
#
#d_iris = np.asarray(dataset[:, :-1], dtype=np.float64)
#label = np.asarray(dataset[:, -1], dtype=np.float64)
#
#pc_cod, d_cod, mean_cod = cdf.pca_cod(d_iris, 2, cod_type='minimum')
#pc_cov, d_cov, mean_cov = cdf.pca_cov(d_iris, 2)
#
#with plt.style.context('seaborn-whitegrid'):
#    plt.figure(figsize=(6,4))
#    plt.title('HBK-data: COD PCA')
#    plt.plot(d_cod[label==0, 0], d_cod[label==0, 1], 'bo', label='setosa')
#    plt.plot(d_cod[label==1, 0], d_cod[label==1, 1], 'ro', label='versicolor')
#    plt.plot(d_cod[label==2, 0], d_cod[label==2, 1], 'go', label='virginica')
#    plt.legend(loc='lower center')
#    plt.tight_layout()
#    #plot_xyaxis([-40, 40], [-35, 30])
#
#with plt.style.context('seaborn-whitegrid'):
#    plt.figure(figsize=(6,4))
#    plt.title('HBK-data: COV PCA')
#    plt.plot(d_cov[label==0, 0], d_cov[label==0, 1], 'bo', label='setosa')
#    plt.plot(d_cov[label==1, 0], d_cov[label==1, 1], 'ro', label='versicolor')
#    plt.plot(d_cov[label==2, 0], d_cov[label==2, 1], 'go', label='virginica')
#    plt.legend(loc='lower center')
#    plt.tight_layout()
##    plot_xyaxis([-40, 40], [-35, 30])

#d_cod_ = d_cod + mean_cod
#d_cov_ = d_cov + mean_cov

#plt.figure(figsize=(6, 6))
#plt.title('Compare')
#xr = np.array([4.5, 8.5])
#yr = np.array([-8, 8])
#plt.plot(d_iris[:, 0], d_iris[:, 1], 'bo', label='Iris-virginica')
##plot_xyaxis(xr, yr)
#plot_line(pc_cov, mean_cov, xr, set_line='r--', label_='Cov')
#plot_line(pc_cod, mean_cod, xr, set_line='g--', label_='Cod')
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# =========================== 
# HBK data

loc_hbk = 'D:/E/Research_USA/dataset/dataset_hbk/hbk.xlsx'
sheet_name = 'hbk'
df = pd.read_excel(loc_hbk, sheetname=sheet_name, header=0)
dataset = df.values

d_hbk = dataset
pc_cod, d_cod, mean_cod = cdf.pca_cod(d_hbk, 2, cod_type='minimum')
pc_cov, d_cov, mean_cov = cdf.pca_cov(d_hbk, 2)

plt.figure(figsize=(6,4))
plt.title('HBK-data: COD PCA')
plt.plot(d_cod[:, 0], d_cod[:, 1], 'bo')
plot_xyaxis([-40, 40], [-35, 30])

s = 7.378 # 0.975 tolerance ellipse
delta = cdf.matrix_codiff(d_cod.T, cod_type='minimum')
theta = np.arange(0, 2*np.pi, 0.01)
ex = np.sqrt(s * delta[0, 0]) * np.cos(theta)
ey = np.sqrt(s * delta[1, 1]) * np.sin(theta)
plt.plot(ex, ey, 'k-')


plt.figure(figsize=(6,4))
plt.title('HBK-data: COV PCA')
plt.plot(d_cov[:, 0], d_cov[:, 1], 'bo')
plot_xyaxis([-40, 40], [-35, 30])

s = 7.378 # 0.975 tolerance ellipse
delta = np.cov(d_cov.T)
theta = np.arange(0, 2*np.pi, 0.01)
ex = np.sqrt(s * delta[0, 0]) * np.cos(theta)
ey = np.sqrt(s * delta[1, 1]) * np.sin(theta)
plt.plot(ex, ey, 'k-')

#d_cod_ = d_cod + mean_cod
#d_cov_ = d_cov + mean_cov

#plt.figure(figsize=(6,6))
#plt.title('HBK-data')
#xr = np.array([-12, 42.0])
#plt.plot(d_hbk[:, 0], d_hbk[:, 1], 'bo')
#plot_line(pc_cov, mean_cov, xr, set_line='r--', label_='Cov')
#plot_line(pc_cod, mean_cod, xr, set_line='g--', label_='Cod')
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
#plot_xyaxis([-13, 13], [-35, 35])
