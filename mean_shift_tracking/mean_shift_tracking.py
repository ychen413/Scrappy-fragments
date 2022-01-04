# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:55:05 2018

@author: deep362
"""

import numpy as np
import matplotlib.pyplot as plt


def sample_generator(num_sample, area_length=200, dis_type = 1):
    # dis_type 1: Normal distribution
    #          2: Uniform distribution
    
    if dis_type == 1:
        sample = area_length / 2 * np.random.randn(num_sample, 2) / 3
        plt.plot(sample[:, 0], sample[:, 1], 'o')
    elif dis_type == 2:
        sample = area_length * np.random.random_sample((num_sample, 2)) - area_length / 2
        plt.plot(sample[:, 0], sample[:, 1], 'o')
    else:
        print('Please select distribution type 1 or 2')       
        pass
    
    return sample

def sensor_pusher(sensor_loc, sensor_loc_last, step_scale=1):
    # Random step towards up / down / left / right
    # Check the path in order to avoid turning back
    
    direction = np.array([[0, 1.0], [0, -1.0], [1.0, 0], [-1.0, 0]])
    turnback_dir = np.inner((sensor_loc_last - sensor_loc), direction)
    print(turnback_dir)
    mask = turnback_dir != np.max(turnback_dir)
    next_loc = sensor_loc + direction[mask][np.random.choice(len(direction[mask]), 1)].flatten() * step_scale
    return next_loc

#def sensor_pusher_weight(sample_detect, weight_detect):
#    sample_heavy = sample_detect[sample_detect == np.max(weight_detect)]
#    sensor_loc = np.sum(sample_heavy, axis=0) / len(sample_heavy[:,0])
#    return sensor_loc

#def mean_shift_tracking(sample, sensor_loc, sensor_radius, num_step):
#    # 2D problem
#    # sensor_loc is [x, y]
#
#    sensor_track = np.empty((num_step, 2))    
#    for t in np.arange(num_step):
#        sam_detect = sample[np.linalg.norm(sample-sensor_loc, axis=1) < sensor_radius]
#        sensor_loc = np.sum(sam_detect, axis=0) / len(sam_detect[:,0])
#        sensor_track[t, :] = sensor_loc
#
#    return sensor_track
    
#def mean_shift_tracking_modify(sample, sensor_loc, sensor_radius, num_step):
#    # 2D problem
#    # sensor_loc is [x, y] (2, )
#    # Decrease weight of detected samples
#    # If converges, use pusher to force the sensor move
#    
#    weight = np.ones(len(sample))
#    sensor_track = np.empty((num_step+1, 2))
#    sensor_track[0, :] = sensor_loc
#    
#    for t in np.arange(num_step)+1:
#        mask = np.linalg.norm(sample-sensor_loc, axis=1) < sensor_radius
#        sam_detect = sample[mask]
#        sensor_loc = np.sum(weight[mask][:, None] * sam_detect, axis=0) / np.sum(weight[mask])
#        if np.linalg.norm(sensor_track[t-1, :] - sensor_loc) < 1e-2:
#            sensor_loc = sensor_pusher(sensor_loc, sensor_track[t-2, :], 2)
#        
#        weight[mask] *= 0.9
#        weight /= np.linalg.norm(weight)
#        sensor_track[t, :] = sensor_loc
#        print(sensor_loc)
#    
#    return sensor_track

def mean_shift_tracking_kernel(sample, sensor_loc, sensor_radius, num_step):
    # 2D problem
    # sensor_loc is [x, y] (2, )
    # Decrease weight of detected samples
    # If converges, use pusher to force the sensor move
    
    weight = np.ones(len(sample))
    sensor_track = np.empty((num_step+1, 2))
    sensor_track[0, :] = sensor_loc
    
    for t in np.arange(num_step)+1:
        mask = np.linalg.norm(sample-sensor_loc, axis=1) < sensor_radius
        sam_detect = sample[mask]
        # add kernel for every samples inside the sensor
        sam_detect_kernel = (sam_detect[:, None, :] + np.random.normal(0.0, 0.5, (10, 2))[None, :, :]).reshape(-1, 2)
        weight_detect_kernel = np.repeat(weight[mask], 10)
        
        sensor_loc = np.sum(weight_detect_kernel[:, None] * sam_detect_kernel, axis=0) / np.sum(weight_detect_kernel)
        if np.linalg.norm(sensor_track[t-1, :] - sensor_loc) < 1e-2:
            sensor_loc = sensor_pusher(sensor_loc, sensor_track[t-2, :], 2)
        
        weight[mask] *= 0.9
        weight /= np.linalg.norm(weight)
        sensor_track[t, :] = sensor_loc
        print(sensor_loc)
    
    return sensor_track

def spiral_tracking(sample, sensor_loc, sensor_radius, delta_theta=10):
    # Counter-clockwise til center
    # Use polar coodinates to rotate
    # delta_r is the gap between the two circles
    # delta_theta is the sampling period (deg)
    r = np.linalg.norm(sensor_loc)
    theta = np.arctan2(sensor_loc[1], sensor_loc[0])
    delta_r = 1.5 * sensor_radius
    
    num_circle = int(np.floor(r / delta_r))
    sample_theta = np.tile((np.arange(0, 360, delta_theta) + theta), num_circle)
    sample_r = np.linspace(r, 0, len(sample_theta))
    
    sensor_track = np.vstack(((sample_r * np.cos(sample_theta*np.pi/180.0)), \
                               (sample_r * np.sin(sample_theta*np.pi/180.0)))).transpose()
    
    return sensor_track
    
    
def evaluate_num_samples_detect(sample, sensor_trace, sensor_radius):
    dis = np.linalg.norm(sample[None, :, :]-sensor_trace[:, None, :], axis=2)
    mask = dis < sensor_radius
    
    count = dis
    count[:] = 0
    count[mask==True] = 1
    count[1:, :][count[1:, :] + count[:-1, :]==2] = 0
    return np.sum(count)
    
def tracking_distance(sensor_track):
    return np.sum(np.linalg.norm(sensor_track[1:, :] - sensor_track[:-1, :], axis=1))

#def draw_sensor(sensor_loc, sensor_radius, color_):
#    plt.plot(sensor_loc[0], sensor_loc[1], 'ro')
#    circle = plt.Circle((sensor_loc[0], sensor_loc[1]), sensor_radius, color=color_, fill=False)
#    ax = plt.gca()
#    ax.add_artist(circle)
#    pass

