# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 02:32:25 2019

@author: deep362
"""

import numpy as np
import matplotlib.pyplot as plt
import KPS_function as kf
import pandas as pd

def gen_data(ind):
    np.random.seed(ind)
    group_0 = kf.sample_gen(0, [-2.0, 2.0], num_sample=200, len_x1=8, len_x2=5, tilt=30)
    group_1 = kf.sample_gen(1, [2.0, -2.0], num_sample=200, len_x1=8, len_x2=5, tilt=30)
    return group_0, group_1

def make_test_data(dataset, test_rate=0.2):
    len_dataset = np.size(dataset, axis=0)
    num_test_data = int(np.floor(len_dataset * test_rate))
    ind_test_data = np.random.choice(len_dataset, num_test_data)
    testing_data = dataset[ind_test_data, :-1]
    answer = dataset[ind_test_data, -1]
    training_data = dataset[np.setdiff1d(np.arange(len_dataset), ind_test_data), :]
    return training_data, testing_data, answer

def sample_gen_ring(label, num_sample, center, r_max, r_min):
    # Generate samples in a ring region with radius [r_max, r_min]
    # Generate samples in a circle [r_max] if set r_min = 0
    # Output: group [x1, x2, label] (m, 3)
    np.random.seed(7)
    theta = 2 * np.pi * np.random.random_sample(num_sample)
    r = (r_max - r_min) * np.random.random_sample(num_sample) + r_min
    sample = np.array([r * np.cos(theta), r * np.sin(theta)]).transpose()
    group = np.concatenate((sample + center, np.repeat(label, num_sample).reshape(-1, 1)), axis=1)
    return group

# ===================== self-generate data ======================
#g0, g1 = gen_data(5)
#dataset = np.concatenate((g0, g1), axis=0)
#d_tr, d_te, ans = make_test_data(dataset, test_rate=0.2) 
#
#plt.figure()
#kf.sample_plot(dataset)
##plt.plot(d_tr[:, 0], d_tr[:, 1], 'bo', label = 'training data')
#plt.plot(d_te[:, 0], d_te[:, 1], 'mD', label = 'testing data')
#plt.title('Cross Validation of the dataset')
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
#
## K-NN:
#pred_knn = kf.knn_multipoint(d_te, d_tr, 5)
#accuracy_knn = np.sum(pred_knn == ans) / np.size(ans) * 100.0
#print('K-NN (K=1):%1.2f%%' % accuracy_knn)
#
## K-PS:
#kp = 2
#pp_m, group_fin = kf.kps_multipoint(d_tr, num_ps=kp)
#plt.figure()
#kf.sample_plot(dataset)
#kf.kps_plot_path(pp_m)
#kf.kps_plot_decision_points(pp_m)
#
#plt.plot(d_te[:, 0], d_te[:, 1], 'mD', label = "test_data")
#pred_kps = kf.kps_classifier(d_te, pp_m, group_fin)
#accuracy_kps = np.sum(pred_kps == ans) / np.size(ans) * 100.0
#print('K-PS (K=5):%1.2f%%' % accuracy_kps)
#plt.title('K-PS (K=%d)' % kp)
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# ==================== Iris data =============================
loc_iris = 'D:/E/Research_USA/dataset/dataset_Iris/iris.xlsx'
sheet_name = 'iris'
df = pd.read_excel(loc_iris, sheetname=sheet_name, header=None)
dataset = df.values  

mask0 = np.logical_or(dataset[:, -1]=='Iris-setosa', dataset[:, -1]=='Iris-virginica')
mask1 = np.logical_or(dataset[:, -1]=='Iris-versicolor', dataset[:, -1]=='Iris-setosa')
mask2 = np.logical_or(dataset[:, -1]=='Iris-versicolor', dataset[:, -1]=='Iris-virginica')

d_iris = dataset[mask2, :]
#d_iris[d_iris[:, -1]=='Iris-setosa', -1] = 0
d_iris[d_iris[:, -1]=='Iris-versicolor', -1] = 0
d_iris[d_iris[:, -1]=='Iris-virginica', -1] = 1

d_iris = np.asarray(d_iris, dtype=np.float64)

# LOOCV
num_sample = np.size(d_iris, axis=0)
pred_knn_iris = np.empty(num_sample) 
pred_kps_iris = np.empty(num_sample)
pred_kps_iris_nn = np.empty(num_sample)

for line in np.arange(num_sample):
    d_te = d_iris[line, :-1]
    d_tr = np.delete(d_iris, line, axis=0)
    
    # K-NN (K=1)
#    pred_knn_iris[line], temp = kf.knn(d_te, d_tr, 1)
    
    # K-PS 
    kp = 20
    pp_m, group_fin = kf.kps_multipoint(d_tr, num_ps=kp)
    pred_kps_iris[line] = kf.kps_classifier(np.array([d_te]), pp_m, group_fin)
    pred_kps_iris_nn[line] = kf.kps_classifier_nn(np.array([d_te]), pp_m, group_fin, k=5)
    
accuracy_knn_iris = np.sum(pred_knn_iris == d_iris[:, -1]) / num_sample * 100
accuracy_kps_iris = np.sum(pred_kps_iris == d_iris[:, -1]) / num_sample * 100
accuracy_kps_iris_nn = np.sum(pred_kps_iris_nn == d_iris[:, -1]) / num_sample * 100
print('Accuracy:\n K-NN=%1.2f\n K-PS=%1.2f\n K-PS-NN=%1.2f' \
      % (accuracy_knn_iris, accuracy_kps_iris, accuracy_kps_iris_nn))

# Confusion matrix
p_ver = d_iris[:, -1][pred_kps_iris==0]
p_vir = d_iris[:, -1][pred_kps_iris==1]
ee = np.sum(p_ver==0)
ei = np.sum(p_ver==1)
ie = np.sum(p_vir==0)
ii = np.sum(p_vir==1)
print('ee:', ee, 'ei:', ei, '\nie:', ie, 'ii:', ii)

plt.figure()
plt.plot(d_iris[d_iris[:, -1]==1, 0], d_iris[d_iris[:, -1]==1, 1], 'ro', label='Virginica')
plt.plot(d_iris[d_iris[:, -1]==0, 0], d_iris[d_iris[:, -1]==0, 1], 'bx', label='Versicolor')
plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
kp = 40
pp_m, group_fin = kf.kps_multipoint(d_iris[:, [0, 1, 4]], num_ps=kp)
kf.kps_plot_decision_points(pp_m)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
plt.title('K=%d' % kp)

# ================== Kernel example data ==================
#num_data = 100
#r_in = 0.5
#r_min = 0.7
#r_max = 1.0
#center = np.array([0, 0])
#
#sample_circle = sample_gen_ring(0, num_data, center, r_in, 0)
#sample_ring = sample_gen_ring(1, num_data, center, r_max, r_min)
#data_kernel = np.concatenate((sample_circle, sample_ring))
#
#plt.figure()
#plt.plot(sample_circle[:, 0], sample_circle[:, 1], 'bo', label='group 0')
#plt.plot(sample_ring[:, 0], sample_ring[:, 1], 'rx', label='group 1')
#
## LOOCV
#num_sample = np.size(data_kernel, axis=0)
#pred_knn_kernel = np.empty(num_sample) 
#pred_kps_kernel = np.empty(num_sample)
#pred_kps_kernel_nn = np.empty(num_sample)
#
#kp = 20
#pp_m, group_fin = kf.kps_multipoint(data_kernel, num_ps=kp)
#kf.kps_plot_decision_points(pp_m)
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)
#
#for line in np.arange(num_sample):
#    d_te = data_kernel[line, :-1]
#    d_tr = np.delete(data_kernel, line, axis=0)
#    # K-NN (K=1)
#    pred_knn_kernel[line], temp = kf.knn(d_te, d_tr, 1)
#    # K-PS
#    kp = 20
#    pp_m, group_fin = kf.kps_multipoint(data_kernel, num_ps=kp)
#    pred_kps_kernel[line] = kf.kps_classifier(np.array([d_te]), pp_m, group_fin)
#    pred_kps_kernel_nn[line] = kf.kps_classifier_nn(np.array([d_te]), pp_m, group_fin)
#    
#accuracy_knn_kernel = np.sum(pred_knn_kernel == data_kernel[:, -1]) / num_sample * 100
#accuracy_kps_kernel = np.sum(pred_kps_kernel == data_kernel[:, -1]) / num_sample * 100
#accuracy_kps_kernel_nn = np.sum(pred_kps_kernel_nn == data_kernel[:, -1]) / num_sample * 100
#
#print('Kernel Accuracy:\n K-NN=%1.2f\n K-PS=%1.2f\n K-PS-NN=%1.2f' \
#      % (accuracy_knn_kernel, accuracy_kps_kernel, accuracy_kps_kernel_nn))
#
#
#


