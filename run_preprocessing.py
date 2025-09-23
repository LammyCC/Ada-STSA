# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2024/07/20 23:06
# @Author : caisj
from my_logging import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from config import Storage_Traffic_Flow_Path_Name, Storage_Taxi_Flow_Path_Name, Storage_Taxi_Speed_Path_Name, \
    Storage_CD_Taxi_Flow_Path_Name
from preprocess.preprocessing_data import SensorsTrafficConversion, TaxiTrafficConversion, ChengDuTaxiTrafficConversion
import warnings
from utils.FastText import Pretrain_FastText, UTSL_Fine_Turning
from utils.get_graph import get_CD_trajectory, get_QD_trajectory
from utils.tools import data_smooth, drop_small_flow
warnings.filterwarnings("ignore")


def storage_traffic_flow_speed(path_and_name):
    STC = SensorsTrafficConversion()
    all_traffic_flow = STC.get_all_traffic_flow()
    all_traffic_flow.to_csv(path_and_name, index=False)


def storage_taxi_flow_speed(path_and_name_flow, path_and_name_speed):
    TTC = TaxiTrafficConversion()
    all_taxi_flow, all_taxi_speed = TTC.get_all_taxi_flow_speed()
    all_taxi_flow.to_csv(path_and_name_flow, index=False)
    all_taxi_speed.to_csv(path_and_name_speed, index=False)


def storage_CD_taxi_flow(path_and_name):
    CDTTC = ChengDuTaxiTrafficConversion()
    all_traffic_flow = CDTTC.get_all_traffic_flow()
    all_traffic_flow.to_csv(path_and_name, index=False)


def flow_process(data, flow_threshold, step, city):
    for column_name in tqdm(data.columns.tolist(), desc='Kalman Smoothing'):
        if column_name.isdigit():
            data[column_name] = data_smooth(data[column_name], step)
    data = drop_small_flow(data, flow_threshold)
    if city == 'QD':
        df_location = pd.read_excel(r'./data/node_location/QD_nodes.xlsx')
        location_map = {}
        for index, row in df_location.iterrows():
            location_map[str(row['crossroadID'])] = [row['lng'], row['lat']]
        for name in data.columns.tolist():
            if name.isdigit() and name not in location_map:
                del data[name]
    return data


def time_embedding():
    from datetime import datetime, timedelta

    node_num = 134

    time_embedding = []
    for i in range(1, 15):
        start_date = datetime(2019, 8, i, 7, 5)
        end_date = datetime(2019, 8, i, 19, 0)
        interval = timedelta(minutes=5)
        time_intervals = []
        current_date = start_date
        while current_date <= end_date:
            time_intervals.append(current_date)
            current_date += interval

        # day = [i.day / 30.0 - 0.5 for i in time_intervals]
        hour = [i.hour / 23.0 - 0.5 for i in time_intervals]
        dayofweek = [i.weekday / 6.0 - 0.5 for i in time_intervals]

        # day = np.array([day for _ in range(node_num)]).T # [144,N]
        hour = np.array([hour for _ in range(node_num)]).T # [144,N]
        dayofweek = np.array([dayofweek for _ in range(node_num)]).T # [144,N]
        embedding = np.stack((hour, dayofweek), axis=-1) # [144,N,2]

        for k in range(embedding.shape[0] - 12 - 12 + 1):
            time_slice = embedding[k:k+24, :, :]
            time_embedding.append(time_slice)
    time_embedding = np.array(time_embedding)
    return time_embedding


def save_QD_traj_emb(pretrain_model_save_path, emb_size):
    city = 'QD'
    
    logger.info('Calculating QD embedding...')
    # get_QD_trajectory(get_all_traj=True)
    get_QD_trajectory(get_all_traj=False)

    logger.info('Pretrain QD UTSL...')
    Pretrain_FastText(traj_path=config.Pretrain_save_path_QD, model_save_path=pretrain_model_save_path, emb_size=emb_size, city=city)

    logger.info('Fine Turning QD UTSL...')
    all_emb = UTSL_Fine_Turning(pretrained_model_path=pretrain_model_save_path, traj_path=config.FineTurning_save_path_QD, emb_size=emb_size, city=city)
    np.save('data/traj_emb/QD_emb.npy', all_emb)


def save_CD_traj_emb(pretrain_model_save_path, emb_size, get_all_traj):
    city = 'CD'
    
    logger.info('Calculating CD embedding...')
    get_CD_trajectory(get_all_traj)
    
    logger.info('Pretrain CD UTSL...')
    Pretrain_FastText(traj_path=config.Pretrain_save_path, model_save_path=pretrain_model_save_path, emb_size=emb_size, city=city)
    
    logger.info('Fine Turning CD UTSL...')
    all_emb = UTSL_Fine_Turning(pretrained_model_path=pretrain_model_save_path, traj_path=config.FineTurning_save_path, emb_size=emb_size, city=city)
    np.save('data/traj_emb/CD_emb.npy', all_emb)
    
def sliding_window_sampling(data, window_size=24, samples_per_day=144):
    total_samples, num_features = data.shape
    total_days = total_samples // samples_per_day
    windows_per_day = samples_per_day - window_size + 1
    total_windows = total_days * windows_per_day
    
    windowed_data = np.zeros((total_windows, window_size, num_features))
    
    window_idx = 0
    for day in range(total_days):
        day_start = day * samples_per_day
        day_end = (day + 1) * samples_per_day
        day_data = data[day_start:day_end]
        
        for i in range(windows_per_day):
            window_data = day_data[i:i + window_size]
            windowed_data[window_idx] = window_data
            window_idx += 1
    
    return windowed_data


if __name__ == "__main__":
    # ----------------  step 1：QD flow -------------
    storage_traffic_flow_speed(path_and_name=Storage_Traffic_Flow_Path_Name)

    #----------------  step 2：CD flow -------------
    storage_CD_taxi_flow(path_and_name=Storage_CD_Taxi_Flow_Path_Name)

    #----------------  step 3：flow preprocessing -------------
    df_flow_QD = pd.read_csv(r'data/train_flow/QD_traffic_flow.csv')
    df_flow_QD = df_flow_QD[df_flow_QD['month_day'] < '09-01'].copy()
    df_flow_QD = flow_process(df_flow_QD, config.QD_Flow_Threshold, 144, city='QD')
    df_flow_QD.to_csv(r'data/train_flow/QD_flow.csv', index=False)
    # split train/val/test
    features = df_flow_QD.iloc[:, 1:].values
    train_data = features[:14*144]
    val_data = features[14*144:16*144]
    test_data = features[16*144:]
    train_windowed = sliding_window_sampling(train_data)
    val_windowed = sliding_window_sampling(val_data)
    test_windowed = sliding_window_sampling(test_data)
    np.save(r'data/flow/QD/train_flow.npy', train_windowed)
    np.save(r'data/flow/QD/val_flow.npy', val_windowed)
    np.save(r'data/flow/QD/test_flow.npy', test_windowed)

    df_flow_CD = pd.read_csv(r'data/train_flow/CD_taxi_flow.csv')
    df_flow_CD = flow_process(df_flow_CD, config.CD_Flow_Threshold, 216, city='CD')
    df_flow_CD.to_csv(r'data/train_flow/CD_flow.csv', index=False)
    
    #----------------  step 4：traj preprocessing  ----------------------
    ### CD
    pretrain_model_save_path = 'model/Pretrain'
    emb_size = 100
    get_all_traj = True
    save_CD_traj_emb(pretrain_model_save_path, emb_size, get_all_traj)
    
    # CD
    traj = np.load(r'data/traj_emb/CD_emb.npy')
    traj = traj.reshape(-1, 149, 100)
    train = traj[:2509, :, :]
    val = traj[2509:2509+386, :, :]
    test = traj[2509+386:, :, :]
    np.save(r'data/flow/CD/train_traj.npy', train)
    np.save(r'data/flow/CD/val_traj.npy', val)
    np.save(r'data/flow/CD/test_traj.npy', test)
    
    
    ### QD
    pretrain_model_save_path = 'model/Pretrain'
    emb_size = 100
    save_QD_traj_emb(pretrain_model_save_path, emb_size)
    
    traj = np.load(r'data/traj_emb/QD_emb.npy')
    traj = traj.reshape(-1, 134, 100)
    train = traj[:1694, :, :]
    val = traj[1694:1694+242, :, :]
    test = traj[1694+242:, :, :]
    np.save(r'data/flow/QD/train_traj.npy', train)
    np.save(r'data/flow/QD/val_traj.npy', val)
    np.save(r'data/flow/QD/test_traj.npy', test)

    #----------------  step 5：traj embedding  ----------------------

    logger.info('Qingdao trajectory embedding...')
    key = config.key
    df_flow = pd.read_csv(r'./data/QD_flow_use.csv')
    save_QD_traj_emb(df_flow, key)

    logger.info('Chengdu trajectory embedding...')
    save_CD_traj_emb()

