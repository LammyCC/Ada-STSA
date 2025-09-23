# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/10/29 23:56
# @Author : caisj
# @Email : cai.sj@foxmail.com
# @File : tools.py
# @Software: PyCharm
from math import sin, asin, cos, radians, fabs, sqrt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gensim.models import Word2Vec


def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1, EARTH_RADIUS = 6371.39):
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance

def get_extreme_value(lng, lat, dist=250):
    lat_max = lat + 180 * dist / (6371.39 * 1000 * np.pi)
    lat_min = lat - 180 * dist / (6371.39 * 1000 * np.pi)
    lng_max = lng + 180 * dist / (6371.39 * 1000 * np.pi * cos(radians(lat)))
    lng_min = lng - 180 * dist / (6371.39 * 1000 * np.pi * cos(radians(lat)))
    return lat_max, lat_min, lng_max, lng_min

def data_smooth(x, step):
    from tsmoothie import KalmanSmoother
    smoother = KalmanSmoother(component='level_longseason', component_noise={'level': 0.2, 'longseason': 0.1}, n_longseasons=step)
    smooth_data = smoother.smooth(x).smooth_data.tolist()[0]
    return smooth_data

def drop_small_flow(df_flow, flow_threshold):
    drop_columns = []
    for column_name in df_flow.columns.tolist():
        if column_name.isdigit():
            if df_flow[column_name].mean() < flow_threshold:
                drop_columns.append(column_name)
    df_flow = df_flow.drop(columns=drop_columns)
    return df_flow


def metric_func(pred, y, times):
    result = {}
    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)

    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:, i, :]
        pred_i = pred[:, i, :]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
    return result


def get_mae(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos])).mean()


def get_rmse(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.sqrt(np.square(y_true[non_zero_pos] - y_pred[non_zero_pos]).mean())


def get_mape(y_pred, y_true):
    non_zero_pos = (np.fabs(y_true) > 0.5)
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos]) / y_true[non_zero_pos]).mean()

# 计算节点的词嵌入表示
def get_word_embedding(traj_slice, nodes_encode):
    size = 128
    model = Word2Vec(traj_slice, vector_size=size, window=6, min_count=1, epochs=50, negative=10, sg=1)
    node_embedding = []
    i = 0
    for node in nodes_encode:
        try:
            res = model.wv.get_vector(node)
        except:
            i += 1
            res = [0] * size
        node_embedding.append(res)
    # print(f'-- The {i} nodes is OOV --')
    return node_embedding


if __name__ == "__main__":
    a = get_extreme_value(120.323312031544, 36.0730417809157)
    print(a)