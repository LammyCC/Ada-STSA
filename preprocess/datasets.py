# -*- coding: utf-8 -*-
# @Time : 2023/7/12 14:39
# @Author : Caisj
import numpy as np
import pywt
import torch
from torch.utils.data import Dataset

def wavelet_decomposition(data):
    '''
    时间序列小波分解
    :param data: {'x': [:, T, N], 'y': [:, T, N]}
    :return: data: {'x': [:, T, N, 2], 'y': [:, T, N]}
    '''
    result = []
    for i in range(data['x'].shape[0]):
        temp = data['x'][i]
        temp_res = []
        for k in range(temp.shape[1]):
            coeffs = pywt.wavedec(temp[:, k], wavelet='haar', level=2)
            res = [elem for arr in coeffs for elem in arr]
            temp_res.append(res)
        result.append(temp_res)

    result = np.asarray(result)
    result = np.transpose(result, (0, 2, 1))

    data['x'] = np.stack([data['x'], result], axis=-1)
    return data


class geometric_dataset(Dataset):
    def __init__(self, data_path, traj_path, time_path, node_num, hist_num=12, pred_num=12):
        self.node_num = node_num
        self.hist_num = hist_num
        self.pred_num = pred_num
        # load data
        print("--- load flow ---")
        data = np.load(data_path)
        data = {'x': data[:, :hist_num, :], 'y': data[:, hist_num:, :]}
        # wavelet
        self.data = wavelet_decomposition(data)
        print("--- load traj emb ---")
        self.traj_emb = np.load(traj_path)
        print("--- load time emb ---")
        self.time_embedding = np.load(time_path)

    def fit(self, scaler):
        self.data['x'] = scaler.transform(self.data['x'])
        self.data['y'] = scaler.transform(self.data['y'], axis=0)

    def __getitem__(self, index):
        # flow
        x_enc = torch.tensor(self.data['x'][index, :, :], dtype=torch.float)
        y = torch.tensor(self.data['y'][index, :, :], dtype=torch.float)
        x_dec = torch.cat((x_enc[int(self.hist_num/2):, :, 0].unsqueeze(2), torch.zeros_like(y.unsqueeze(2), dtype=torch.float32)), dim=0)
        # traj emb
        traj_emb = torch.tensor(self.traj_emb[index, :, :], dtype=torch.float)
        # time emb
        x_enc_mark = torch.tensor(self.time_embedding[index, :self.hist_num, :, :], dtype=torch.float) # step:12
        x_dec_mark = torch.tensor(self.time_embedding[index, int(self.hist_num/2):, :, :], dtype=torch.float) # step:18
        return [x_enc, x_enc_mark, x_dec, x_dec_mark, traj_emb, y]

    def __len__(self):
        return self.data['x'].shape[0]
