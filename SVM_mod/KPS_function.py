# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:32:04 2019

@author: deep362
"""

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

def sample_gen(label, center, num_sample=15, len_x1=3.0, len_x2=3.0, tilt=0.0):
    # Generate samples of group [label]
    # The center is at [0, 0] and then move to center(2, )
    # Range: width: -len_x1 to +len_x1; height: -len_x2 to +len_x2
    # The sample will rotate counter-clockwise with tilt (deg) angle when center is at [0, 0]
    
    area = np.array([[len_x1], [len_x2]])
    th = tilt * np.pi / 180.0
    mr = np.array([[np.cos(th), -np.sin(th)], \
                   [np.sin(th), np.cos(th)]])
    sample = np.dot(mr, \
                    area * np.random.random_sample((2, num_sample)) - area / 2).transpose()
    group = np.concatenate((sample + center, \
                            np.repeat(label, len(sample)).reshape(-1, 1)), axis=1)
    return group

def sample_plot(dataset):
    # Input: dataset (M, 3) [x1, x2, label]
    # Maximum num_class is 8
    color = ['b', 'r', 'g', 'y', 'm', 'c', 'k', 'w']
    group = np.unique(dataset[:, -1])
    ind = 0
    for line in group:
        g = dataset[dataset[:, -1]==line, :]    
        plt.plot(g[:, 0], g[:, 1], 'o'+color[ind], label = "group %i" % line)
        ind += 1
        
    pass

def kps_initial_points(group_k, num_points):
    # Input: a (M, N) dataset of one class / group 
    #        #M data, with each [x1, x2, ..., xN] features
    #   Randomly select an initial point
    #   find next point that out of the range of r_main 
    # Output: 'num_points' points in the group k
    g = group_k
    w = np.ones(np.size(g, axis=0))
    w = w / np.sum(w)
    p_ = g[np.random.choice(np.size(g, axis=0), 1, p=w), :]
    p0 = p_  
    
    gx = np.max(g, axis=0) - np.min(g, axis=0)
    r_main = np.max(gx) / num_points
    for line in np.arange(num_points-1):
        w[np.linalg.norm(g -  p_, axis=1) < r_main] = 0
        w = w / np.sum(w)
        p_ = g[np.random.choice(np.size(g, axis=0), 1, p=w), :]
        p0 = np.concatenate((p0, p_), axis=0)
    
    return p0

def kps(dataset):
    # Input: A dataset (M, k+1) with [x1, x2, ..., xk, label]
    #        xi: feature of each data, total k data in the dataset
    #        Only apply for 2-class problem with label 0 & 1
    #
    # Randomly choose one point from g0
    # Output: A decision pair (one point from each group, totally 2 points)
    
    label = dataset[:, -1]
    g0 = dataset[label==0, :-1]
    g1 = dataset[label==1, :-1]
    
    p0 = g0[np.random.choice(np.size(g0, axis=0), 1), :]
    dist = np.linalg.norm(p0 - g1, axis=1)
    p1 = g1[np.argmin(dist), :]
    d = np.min(dist)
    d_ = 0.0
    pp = p0
    
    while d_ != d:
        pp = np.vstack((pp, p1)) 
        num_g = np.mod(np.size(pp, axis=0), 2)
        if num_g == 0:
            g = g0
        else:
            g = g1
        
        p0 = p1
        d = d_
        dist = np.linalg.norm(p0 - g, axis=1)
        p1 = g[np.argmin(dist), :]
        d_ = np.min(dist)
    
    return pp

def kps_multipoint(dataset, num_ps=2):
    # Choose "num_ps" start points randomly from group_0
    # The start points should seperate as far as possible
    # Output: all the points on the path and the group of final point
    
    label = dataset[:, -1]
    g0 = dataset[label==0, :-1]
    g1 = dataset[label==1, :-1]
    num_feature = np.size(g0, axis=1)
    p0 = kps_initial_points(g0, num_ps)
    dist = np.linalg.norm(p0[None, :, :] - g1[:, None, :], axis=2)
    p1 = g1[np.argmin(dist, axis=0), :]
    d01 = np.min(dist, axis=0)
    d01_ = np.zeros(num_ps)
    pp = p0.reshape(num_ps, 1, num_feature)
    while (d01_ != d01).any():
        pp = np.concatenate((pp, p1[:, None, :]), axis=1) 
        num_g = np.mod(len(pp[0, :, 0]), 2)
        if num_g == 0:
            g = g0
        else:
            g = g1
        
        p0 = p1
        d01 = d01_
        dist = np.linalg.norm(p0[None, :, :] - g[:, None, :], axis=2)
        p1 = g[np.argmin(dist, axis=0), :]
        d01_ = np.min(dist, axis=0)
    
    return pp, np.abs(num_g-1)

def kps_get_decision_points(p_trace, group_fin):
    # Input: (1) The trace of points shifting (m, n, k)
    #            m = #initial points, n = #shifting, k = #features
    #        (2) The class of the last points: 0 or 1 (two classes problem)
    # So the final two points in the trace are a pair of the decision points 
    # Output: Decision points array (m_, k+1) with [x1, ..., xk, label]
    #         m_ <= 2*m (different initial points may converge to the same decision point)
        
    dp = p_trace[:, -2:, :].reshape(-1, np.size(p_trace, axis=-1), order='F')
    g_class = np.array([np.abs(group_fin-1), group_fin]).repeat(np.size(dp, axis=0)/2)
    decision_points = np.concatenate((dp, g_class[:, None]), axis=1)

    return decision_points

def kps_classifier_nn(test_data, pp_m, group_fin, k):
    # Classify unknown objects by k-nearest neighbors
    decision_points = kps_get_decision_points(pp_m, group_fin)
    dist = np.linalg.norm(decision_points[:, :-1][None, :, :] - test_data[:, None, :], axis=2)
#    predict = decision_points[np.argmin(dist, axis=1), -1]
    label_nn = decision_points[np.argsort(dist)[:, 0:k], -1]
    predict = np.argmax(np.vstack((np.count_nonzero(label_nn==0, axis=1), \
                         np.count_nonzero(label_nn==1, axis=1))), axis=0)
#    print(label_nn)
#    print(predict)
    return predict

def kps_classifier_dp(test_data, pp_m, group_fin, k):
    # Classify unknown objects by k- decision pairs (k < #initial points)
    # Similiar as knn, but now the voting unit is pair
    # Each pair votes to the class of the shorter distance point
    g_class = np.array([np.abs(group_fin-1), group_fin])
    decision_0 = pp_m[:, -2, :]
    decision_1 = pp_m[:, -1, :]
    
    d0 = np.linalg.norm(decision_0[None, :, :] - test_data[:, None, :], axis=2)
    d1 = np.linalg.norm(decision_1[None, :, :] - test_data[:, None, :], axis=2)
    
    dp = np.concatenate((d0[None, :, :], d1[None, :, :]))
    vote_dist = np.min(dp, axis=0)
    np_ = np.argsort(vote_dist)
    vote = np.argmin(dp, axis=0)
#    predict = np.argmax()
    print(d0.shape, d1.shape, vote)
    pass

def kps_classifier(test_data, pp_m, group_fin):
    # Classify unknown data by comparing the sum of distance to all decision points
    g_class = np.array([np.abs(group_fin-1), group_fin])
    decision_0 = pp_m[:, -2, :]
    decision_1 = pp_m[:, -1, :]
    
    d0 = np.sum(np.linalg.norm(decision_0[None, :, :] - test_data[:, None, :], axis=2), axis=1)
    d1 = np.sum(np.linalg.norm(decision_1[None, :, :] - test_data[:, None, :], axis=2), axis=1)

    predict = g_class[np.argmin(np.vstack((d0, d1)).transpose(), axis=1)]
#    print("K-PS: group", predict)
    
    return predict

def kps_plot_path(points_trace):
    # Plot the initial points and the trace to decision points
    if len(points_trace.shape) == 2:
        points_trace = points_trace[None, :, :]
        
    plt.plot(points_trace[:, 0, 0], points_trace[:, 0, 1], 'kx', \
             markersize = 14, label = "start points")
    
    for line in np.arange(np.size(points_trace, axis=0)):
        plt.plot(points_trace[line, :, 0], points_trace[line, :, 1], 'k--')
    
    pass
    
def kps_plot_decision_points(points_trace):
    # Mark decision points
    if len(points_trace.shape) == 2:
        points_trace = points_trace[None, :, :]
        
    plt.plot(points_trace[:, -2:, 0], points_trace[:, -2:, 1], 'go', \
             markersize=15, markerfacecolor='none', label = "Decision points")
    pass

def make_decision_boundary(pp_m):
    # Temporary not used
    # Make an decision boundary by connecting all the middle points of each decision pair
    p_margin = pp_m[:, -2:, :]
    p_db = np.unique(np.sum(p_margin, axis = 1) / 2, axis=0)
    
    return p_db

def plot_bisection(point1, point2, y1=.5, y2=3.5):
    # Temporary not used
    # Plot the bisection of each decision pair
    w = point2 - point1
    mp = (point2 + point1) / 2.0
    b = -np.sum(w * mp)
    x1 = -(w[1] * y1 + b) / w[0]
    x2 = -(w[1] * y2 + b) / w[0]
    plt.plot([x1, x2], [y1, y2], 'g--')
    pass
    
def knn(test_point, dataset, k):
    data = dataset[:, :-1]
    label = dataset[:, -1]
    
    dist = np.linalg.norm(data - test_point, axis=1)
    label_nn = label[np.argsort(dist)[0:k]]
    if np.count_nonzero(label_nn==0) > np.count_nonzero(label_nn==1):
        g_tp = 0
    else:
        g_tp = 1
    
#    print("K-NN: group", g_tp)
    knn_point = data[np.argsort(dist)[0:k]]
    
    return g_tp, knn_point

def knn_multipoint(test_points, data, k):
    dataset = data[:, 0:2]
    label = data[:, -1]
    
    dist = np.linalg.norm(dataset[None, :, :] - test_points[:, None, :], axis=2)
    index_knn = np.argsort(dist)[:, 0:k]  # index of k nearest neighbors of each test point
    label_nn = np.take(label, index_knn)
    
    predict = np.zeros(np.size(label_nn, axis=0))
    mask = np.count_nonzero(label_nn==0, axis=1) < np.count_nonzero(label_nn==1, axis=1)
    predict[mask] = 1
    
    print("K-NN: group", predict)
#    knn_points = np.take(dataset, index_knn)
    
    return predict

