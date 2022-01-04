# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:10:47 2019

@author: deep362
"""

import numpy as np
import matplotlib.pyplot as plt
import KPS_function as kf
import pandas as pd

def test_data_01():
    group_0 = kf.sample_gen(0, [2, 2])
    group_1 = kf.sample_gen(1, [-2, 2])
    return group_0, group_1

def test_data_02():
    group_0 = np.array([
            [2.978800076189068058e+00, 1.763061485775478499e+00, 0],
            [2.366301678381228069e+00, 2.891592294311696065e+00, 0],
            [1.321325178959200652e+00, 1.791620551115567839e+00, 0],
            [2.743530598543006960e+00, 5.796699016167616847e-01, 0],
            [3.488928220601144048e+00, 2.606762184254467929e+00, 0],
            [1.172261403820892944e+00, 2.098829166557472536e+00, 0],
            [5.363379093384030849e-01, 1.065064704282756125e+00, 0],
            [2.729254351951896584e+00, 1.461429675314203980e+00, 0],
            [1.030887324661305282e+00, 2.567870705048071578e+00, 0],
            [2.600939744639764406e+00, 2.608081547891303309e+00, 0],
            [2.341062414157422999e+00, 3.248941397314891955e+00, 0],
            [3.067980584245228126e+00, 1.155842753099434628e+00, 0],
            [5.411724564057203146e-01, 3.251933256476054801e+00, 0],
            [1.476762089173393910e+00, 1.338843199200761891e+00, 0],
            [2.267985765242864549e+00, 2.100350686446053494e+00, 0],
            ])
    
    group_1 = np.array([
            [-2.748114844967075143e+00, 1.723722287237551054e+00, 1],
            [-1.078014497820662498e+00	, 3.196441544951032387e+00, 1],
            [-9.061661322856582323e-01, 5.195999784603895222e-01, 1],
            [-3.291322596887047425e+00	, 3.282155356248157130e+00, 1],
            [-1.885545825730614800e+00	, 1.893270095296983424e+00, 1],
            [-1.806664360328564367e+00	, 9.341013378286371349e-01, 1],
            [-1.810366857511330974e+00	, 2.798287615098343561e+00, 1],
            [-1.022210791304575039e+00	, 1.127024983614901998e+00, 1],
            [-2.116851569047342174e+00	, 9.768453566753572126e-01, 1],
            [-2.698543242244600293e+00	, 1.201665307525396242e+00, 1],
            [-5.806150396014304604e-01	, 3.468182878864008334e+00, 1],
            [-3.358902940152235672e+00	, 2.920027448352747967e+00, 1],
            [-1.469615500412488007e+00	, 2.791155459343710010e+00, 1],
            [-1.504348174147589701e+00	, 2.875635524126816556e+00, 1],
            [-1.707937330594414327e+00	, 2.462697117939763647e+00, 1],
            ])
    return np.concatenate((group_0,group_1))
    
def test_data_import(loc, sheet_name):
    df = pd.read_excel(loc, sheetname=sheet_name)
    dataset = df.values[:, 1:]     
    return dataset

dataset_02 = test_data_02()
# ==============Test: select initial points
#group_0 = dataset_02[dataset_02[:,-1]==0, :]
#group_1 = dataset_02[dataset_02[:,-1]==1, :]
#p0 = kf.kps_initial_points(group_0[:, :-1], 3)
#plt.figure()
#kf.sample_plot(np.concatenate((group_0, group_1), axis=0))
#plt.plot(p0[:, 0], p0[:, 1], 'kx', markersize = 14, label = "start points")

# ============== Only one starting point ==================
#pp = kf.kps(dataset_02)
#plt.figure()
#kf.sample_plot(dataset_02)
#kf.kps_plot_path(pp)
#kf.kps_plot_decision_points(pp)
#plt.title('K-PS: Single initial point')
#plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# ============== KPS with multiple starting points ==================
kp = 3
pp_m, group_fin = kf.kps_multipoint(dataset_02, num_ps=kp)
plt.figure()
kf.sample_plot(dataset_02)
kf.kps_plot_path(pp_m)
kf.kps_plot_decision_points(pp_m)
plt.title('K-PS: Multi-initial point (K=%d)' % kp)

# Test data
test_point = np.array([[-0.5, 1.5]])
plt.plot(test_point[:, 0], test_point[:, 1], 'mD', label = "test_data")
g_tp = kf.kps_classifier(test_point, pp_m, group_fin)

plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# ============== K-NN: Single input ================
k = 3
g_tp_knn, knn_point = kf.knn(test_point, dataset_02, k)

plt.figure()
kf.sample_plot(dataset_02)
plt.plot(test_point[:, 0], test_point[:, 1], 'mD', label = "test_data")
plt.plot(knn_point[:, 0], knn_point[:, 1], 'go', \
         markersize=15, markerfacecolor='none', label = "K-NN")
#plt.scatter(knn_point[:, 0], knn_point[:, 1], s=150, marker='o', facecolors='none', edgecolors='g', label = "K-NN")
plt.title('K-NN: Single input (K=%d)' % k)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# ============== K-NN: multi-input ===================
test_points = np.array([[-0.5, 1.5], [0.5, 2], [-1,0.5]])
k = 3
predict = kf.knn_multipoint(test_points, dataset_02, k)

plt.figure()
kf.sample_plot(dataset_02)
plt.plot(test_points[:, 0], test_points[:, 1], 'mD', label = "test_data")
plt.title('K-NN: Multi-input (K=%d)' % k)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)

# ============= K-NN for HW5 of ECE407 (test_data_03) ================
loc = 'D:/Dropbox/UIC/Course/Q1-data.xlsx'
sheet_name = 'Iris'
dataset = test_data_import(loc, sheet_name)
p_q1 = np.array([[5.65, 3.4], [4.9, 2.7]])

pred_knn = kf.knn_multipoint(p_q1, dataset, 3)
pp_m_d, group_fin_d = kf.kps_multipoint(dataset, num_ps = 3)
prid_ksp = kf.kps_classifier(p_q1, pp_m_d, group_fin_d)
prid_ksp_nn = kf.kps_classifier_nn(p_q1, pp_m_d, group_fin_d, 3)
print(prid_ksp, prid_ksp_nn)

plt.figure()
kf.sample_plot(dataset)
plt.plot(p_q1[:, 0], p_q1[:, 1], 'mD', label = "test_data")
kf.kps_plot_path(pp_m_d)
kf.kps_plot_decision_points(pp_m_d)
plt.legend(bbox_to_anchor=(1.02, 1.0), loc=2, borderaxespad=0.)