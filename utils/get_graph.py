# -*- coding: utf-8 -*-
# @Time : 2023/6/20 19:45
# @Author : Caisj
from datetime import datetime
from scipy import spatial
import requests
import transbigdata
from scipy.spatial.distance import euclidean
import math
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import config
from my_logging import logger
from preprocess.preprocessing_data import ChengDuTaxiTrafficConversion, SensorsTrafficConversion, \
    multiprocessing_change_coordinate, multiprocessing_geohash
from utils.FastText import Pretrain_FastText
from utils.multiprocessing import run_task_multiprocessing
from itertools import groupby
import osmnx as ox
from utils.tools import get_word_embedding


def remove_consecutive_duplicates(df, vehicle_id):
    result = []
    for vin, group in df.groupby(vehicle_id):
        grid_sequence = group['grid'].tolist()
        deduplicated_sequence = [grid_sequence[i] for i in range(len(grid_sequence)) if i == 0 or grid_sequence[i] != grid_sequence[i-1]]
        result.append(deduplicated_sequence)
    
    return result


#####################################   QD traj preprocessing   ####################################
def get_QD_trajectory(get_all_traj, city='QD'):
    logger.info('----- QD: The historical sequences of Qingdao taxis were extracted and encoded... -----')
    STC = SensorsTrafficConversion()

    pretrain_traj = []
    day = 0
    for name in tqdm(os.listdir(config.Train_Taxi_Path), desc=f'{city} trajectory data processing...'):
        df_trajectory = pd.read_csv(os.path.join(config.Train_Taxi_Path, name))
        df_count = df_trajectory.groupby(['taxiID']).size().reset_index()
        df_count = df_count.rename(columns={0: 'num'})
        taxi_id = df_count[(df_count['num'] < 2500) & (df_count['num'] > 500)]['taxiID'].tolist()
        df_trajectory = df_trajectory[df_trajectory['taxiID'].isin(taxi_id)]
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=4, method=STC.multiprocessing_change_coordinate)
        df_trajectory = transbigdata.clean_outofbounds(df_trajectory, bounds=config.QD_Bounds, col=['lng', 'lat'])
        df_trajectory = df_trajectory.sort_values(by=['taxiID', 'timestamp']).reset_index(drop=True)
        df_trajectory = transbigdata.traj_clean_drift(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'], method='twoside',
                                                      speedlimit=80, dislimit=1000, anglelimit=30)
        # df_trajectory = transbigdata.traj_densify(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'], timegap=15)
        df_trajectory = transbigdata.traj_clean_redundant(df_trajectory, col=['taxiID', 'timestamp', 'lng', 'lat'])
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=4, method=STC.multiprocessing_geometry_map_matching)
        df_trajectory = df_trajectory.sort_values(by=['taxiID', 'timestamp']).reset_index(drop=True)
        
        logger.info('GeoHash...')
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=8, method=multiprocessing_geohash)
        
        if get_all_traj:
            trajectory_sequence = remove_consecutive_duplicates(df_trajectory, vehicle_id='taxiID')
            pretrain_traj.extend(trajectory_sequence)
            logger.info('Pretrain data Save...')
            day += 1
            with open(config.Pretrain_save_path_QD + f'/{city}_pretrain_data_{day}.txt', 'w') as f:
                for row in pretrain_traj:
                    f.write(' '.join(row) + '\n')
        
        else:
            logger.info('Get Fine Turning Data...')
            df_trajectory["timestamp"] = pd.to_datetime(df_trajectory["timestamp"])
            file_date = datetime.strptime(str(df_trajectory["timestamp"].min())[:11] + "06:59:59", "%Y-%m-%d %H:%M:%S")
            df_trajectory["time_slice"] = ((df_trajectory["timestamp"] - file_date).dt.seconds // 300 + 1)
            grouped = df_trajectory.groupby(['taxiID', 'time_slice'])['grid'].apply(list).reset_index()
            day_traj = []
            max_time_slice = 145
            for i in range(1, max_time_slice):
                start_slice = max(1, i - 11)
                end_slice = i
                temp = grouped[(grouped['time_slice'] >= start_slice) & (grouped['time_slice'] <= end_slice)]
                trajectory_res = (
                    temp.groupby('taxiID')['grid']
                    .apply(lambda x: [grid for sublist in x for grid in sublist])
                    .tolist()
                )
                trajectory_res = [[k for k, _ in groupby(traj)] for traj in trajectory_res]
                trajectory_res = [traj for traj in trajectory_res if len(traj) >= 3]

                day_traj.append(trajectory_res)
            
            day += 1
            logger.info('Fine Turning Data Save...')
            with open(config.FineTurning_save_path_QD + f'/{city}_fine_turning_data_{day}.pkl', 'wb') as f:
                pickle.dump(day_traj, f)





#####################################   CD traj preprocessing   ####################################
def get_CD_trajectory(get_all_traj, city='CD'):
    logger.info('----- CD: The historical sequences of Chengdu taxis were extracted and encoded... -----')

    CDTTC = ChengDuTaxiTrafficConversion()
    pretrain_traj = []
    day = 0
    for name in tqdm(os.listdir(config.CD_Taxi_Path), desc=f'{city} trajectory data processing...'):
        if '.pkl' not in name:
            continue
        df_trajectory = pd.read_pickle(os.path.join(config.CD_Taxi_Path, name))
        logger.info('Range filtering...')
        df_trajectory = transbigdata.clean_outofbounds(df_trajectory, bounds=config.CD_Bounds, col=['lng', 'lat'])
        logger.info('Trajectory ordering...')
        df_trajectory = df_trajectory.sort_values(by=['vin', 'time']).reset_index(drop=True)
        logger.info('Drift point filtering...')
        df_trajectory = transbigdata.traj_clean_drift(df_trajectory, col=['vin', 'time', 'lng', 'lat'], method='twoside',
                                                      speedlimit=80, dislimit=1000, anglelimit=30)
        logger.info('Instantaneous change filtering...')
        df_trajectory = transbigdata.clean_taxi_status(df_trajectory, col=['vin', 'time', 'status'], timelimit=None)
        logger.info('Repeat position filtering...')
        df_trajectory = transbigdata.traj_clean_redundant(df_trajectory, col=['vin', 'time', 'lng', 'lat'])
        logger.info('Coordinate conversion...')
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=8, method=multiprocessing_change_coordinate)
        logger.info('Map matching...')
        data_deliver = data_deliver.drop(columns=['status'])  # 减小内存
        data_res = pd.DataFrame()
        # 切分成4段计算，防止内存溢出
        slice_num = 4
        for i in tqdm(range(0, slice_num)):
            if i == 0:
                continue
            temp = data_deliver.iloc[i * math.ceil(len(data_deliver) / slice_num): (i + 1) * math.ceil(len(data_deliver) / slice_num)]
            temp = run_task_multiprocessing(data=temp, threads=12, method=CDTTC.multiprocessing_geometry_map_matching)
            data_res = data_res.append(temp).reset_index(drop=True)
        data_deliver = data_res
        del data_res
        G = ox.load_graphml(r'./data/roadnet/chengdu.graphml')
        data_deliver = transbigdata.traj_mapmatch(data_deliver, G, col=['lng', 'lat'])
        
        logger.info('GeoHash...')
        # df_trajectory['grid'] = transbigdata.geohash_encode(df_trajectory['lng'], df_trajectory['lat'], precision=6)
        df_trajectory = run_task_multiprocessing(data=df_trajectory, threads=8, method=multiprocessing_geohash)
        
        if get_all_traj:
            trajectory_sequence = remove_consecutive_duplicates(df_trajectory, vehicle_id='vin')
            pretrain_traj.extend(trajectory_sequence)
            logger.info('Pretrain data Save...')
            day += 1
            with open(config.Pretrain_save_path + f'/{city}_pretrain_data_{day}.txt', 'w') as f:
                for row in pretrain_traj:
                    f.write(' '.join(row) + '\n')
            # ##### 预训练UTSL
            # logger.info('Pretrain UTSL Model...')
            # Pretrain_FastText(f'data/traj_emb/{city}_pretrain_data.txt', pretrain_model_save_path, emb_size, city)
        
        else:
            logger.info('Get Fine Turning Data...')
            df_trajectory["time"] = pd.to_datetime(df_trajectory["time"])
            file_date = datetime.strptime(str(df_trajectory["time"].min())[:11] + "05:59:59", "%Y-%m-%d %H:%M:%S")
            df_trajectory["time_slice"] = ((df_trajectory["time"] - file_date).dt.seconds // 300 + 1)
            grouped = df_trajectory.groupby(['vin', 'time_slice'])['grid'].apply(list).reset_index()
            day_traj = []
            max_time_slice = 217
            for i in range(1, max_time_slice):
                start_slice = max(1, i - 11)
                end_slice = i
                temp = grouped[(grouped['time_slice'] >= start_slice) & (grouped['time_slice'] <= end_slice)]

                trajectory_res = (
                    temp.groupby('vin')['grid']
                    .apply(lambda x: [grid for sublist in x for grid in sublist])
                    .tolist()
                )
                trajectory_res = [[k for k, _ in groupby(traj)] for traj in trajectory_res]
                trajectory_res = [traj for traj in trajectory_res if len(traj) >= 3]

                day_traj.append(trajectory_res)
            
            day += 1
            logger.info('Fine Turning Data Save...')
            with open(config.FineTurning_save_path + f'/{city}_fine_turning_data_{day}.pkl', 'wb') as f:
                pickle.dump(day_traj, f)


