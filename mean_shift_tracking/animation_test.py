# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 02:47:05 2018

@author: deep362
"""

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np
import mean_shift_tracking as mst

# Generate samples in the area needed detect
edge_length = 200
sample = mst.sample_generator(1000, edge_length, 1)
mean_exact = np.sum(sample, axis=0) / len(sample[:, 0])

# Sensor's properties and initial location
sensor_loc = np.array([80.0, -10.0])
sensor_r = 10.0
step = 100
#sensor_track = mst.mean_shift_tracking(sample, sensor_loc, sensor_r, step)
#sensor_track_m = mst.mean_shift_tracking_modify(sample, sensor_loc, sensor_r, step)
sensor_track_k = mst.mean_shift_tracking_kernel(sample, sensor_loc, sensor_r, step)
sensor_track_s = mst.spiral_tracking(sample, sensor_loc, sensor_r)

# Plot the trace on the samples map
fig, ax = plt.subplots(figsize=(10, 8))
ax.set(xlim=(-100, 100), ylim=(-100, 100))
#plt.plot(sensor_track[:, 0], sensor_track[:, 1], 'bo')
plt.plot(sample[:, 0], sample[:, 1], 'go')
plt.plot(sensor_track_k[:, 0], sensor_track_k[:, 1], 'ro')
plt.plot(mean_exact[0], mean_exact[1], 'kx')
plt.plot(sensor_track_s[:, 0], sensor_track_s[:, 1], 'bo')

# Evaluate comparing parameter
# mean shift tracking:
#count_m = mst.evaluate_num_samples_detect(sample, sensor_track_m, sensor_r)
#total_dis_m = mst.tracking_distance(sensor_track_m)
#count_m_dis = count_m / total_dis_m
#print('\nMean shift tracking:')
#print('total detected sample:', count_m)
#print('total distance:', total_dis_m)
#print('efficiency:', count_m_dis)
#print('\n')

# spiral trace:
count_s = mst.evaluate_num_samples_detect(sample, sensor_track_s, sensor_r)
total_dis_s = mst.tracking_distance(sensor_track_s)
count_s_dis = count_s / total_dis_s
print('Spiral trace:')
print('total detected sample:', count_s)
print('total distance:', total_dis_s)
print('efficiency:', count_s_dis)

# mean shift tracking:
count_k = mst.evaluate_num_samples_detect(sample, sensor_track_k, sensor_r)
total_dis_k = mst.tracking_distance(sensor_track_k)
count_k_dis = count_k / total_dis_k
print('\nMean shift tracking:')
print('total detected sample:', count_k)
print('total distance:', total_dis_k)
print('efficiency:', count_k_dis)
print('\n')