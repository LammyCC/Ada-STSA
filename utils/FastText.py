'''
Author: Caisj
Date: 2024-12-31 15:19:14
LastEditTime: 2025-09-23 14:36:56
'''
import os
import pickle
from gensim.models import FastText
import numpy as np
import pandas as pd
from tqdm import tqdm
from tsmoothie import KalmanSmoother
import transbigdata

def data_smooth(x, step):
    smoother = KalmanSmoother(component='level_longseason', component_noise={'level': 0.25, 'longseason': 0.15}, n_longseasons=step)
    smooth_data = smoother.smooth(x).smooth_data.tolist()[0]
    return smooth_data

def Pretrain_FastText(traj_path, model_save_path, emb_size, city):
    dirs = os.listdir(traj_path)[:8]
    traj = []
    for file in tqdm(dirs, desc='Pretrain data concat progress: '):
        with open(os.path.join(traj_path, file), 'r') as f:
            for line in f:
                row = line.strip().split(' ')
                traj.append(row)
    traj = [i for i in traj if len(i) >= 3]
    embedding_size = emb_size
    window_size = 5
    min_count = 1
    workers = 8
    sg = 0
    negative = 5
    epochs = 10
    
    ##### 预训练
    model = FastText(sentences=traj, 
                     vector_size=embedding_size,
                     window=window_size,
                     min_count=min_count,
                     workers=workers,
                     sg=sg,
                     negative=negative,
                     epochs=epochs
                     )
    
    model.save(model_save_path+f'/{city}_UTSL_{emb_size}.model')
    print(f"Model load: {model_save_path}")


def UTSL_Fine_Turning(pretrained_model_path, traj_path, emb_size, city):
    all_emb = []  # [T,N,C]
    if city == "CD":
        time_step = 216
    else:
        time_step = 144
    for name in tqdm(os.listdir(traj_path)):
        traj_data = pickle.load(open(os.path.join(traj_path, name), 'rb'))
        traj_data = [[sample for sample in traj if len(sample) >= 3] for traj in traj_data]
        day_emb = []  # [T,N,C]
        for i in range(0, time_step-12-12+1):
            traj_current = traj_data[i]
            pretrained_model = FastText.load(pretrained_model_path + f'/{city}_UTSL_{emb_size}.model')
            pretrained_model.build_vocab(traj_current, update=True)
            fine_tuning_epochs = 1
            fine_tuning_alpha = 0.01
            pretrained_model.alpha = fine_tuning_alpha
            pretrained_model.min_alpha = fine_tuning_alpha
            pretrained_model.train(traj_current, total_examples=len(traj_current), epochs=fine_tuning_epochs)
            use = pd.read_csv(f'data/{city}_flow_use.csv')
            if city == "CD":
                loc = pd.read_csv(f'data/node_location/{city}_railway.csv')
                node_id = use.columns[2:]
                loc['node_id'] = loc['node_id'].astype(str)
                loc['grid'] = transbigdata.geohash_encode(loc['lng'], loc['lat'], precision=6)
                grids = []
                for node in node_id:
                    grid = loc[loc['node_id'] == node].iloc[0][['grid']][0]
                    grids.append(grid)
            else:
                loc = pd.read_excel(r'data/node_location/QD_nodes.xlsx')
                node_id = use.columns[1:]
                loc['crossroadID'] = loc['crossroadID'].astype(str)
                loc['grid'] = transbigdata.geohash_encode(loc['lng'], loc['lat'], precision=6)
                grids = []
                for node in node_id:
                    grid = loc[loc['crossroadID'] == node].iloc[0][['grid']][0]
                    grids.append(grid)

            time_emb = []  # [N,C]
            for grid in grids:
                node_emb = pretrained_model.wv[grid]
                time_emb.append(node_emb)
            day_emb.append(time_emb)
        all_emb.append(day_emb)
    return np.array(all_emb)
    
