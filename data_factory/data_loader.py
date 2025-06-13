import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore   # SciPy가 없다면 np.nanmean/np.nanstd로 대체 가능

def detect_outliers_zscore(series, threshold: float = 2):
    """
    1-차원 배열(series)에 대해 z-score 기준 이상치(절대값 > threshold) 마스크 반환
    """
    z = zscore(series, nan_policy="omit")
    return np.abs(z) > threshold


class CryptoSegLoader(object):
    """
    bitcoin_ethereum_transactions.csv 전용 세그먼트 데이터로더
    ---------------------------------------------------------
    · csv 구조: [date, transactions_bitcoin, transactions_ethereum, ...]
    · split 비율: train 60 % / val 20 % / test 20 %
      (비율·threshold는 파라미터로 조정 가능)
    · train split 분포로만 StandardScaler를 fit → 데이터 누수 방지
    · 각 split 내부에서 z-score로 point-level anomaly label 생성
    · __getitem__  → (win_size, num_features) 슬라이스, 동일 길이 레이블
    """

    def __init__(self,
                 csv_path: str,
                 win_size: int,
                 step: int,
                 mode: str = "train",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 z_th: float = 2.0):
        # assert mode in ["train", "val", "test"]
        self.mode, self.win_size, self.step = mode, win_size, step

        # 1) 로드 & 전처리
        df = pd.read_csv(csv_path)
        values = np.nan_to_num(df.iloc[:, 1:].values)        # 첫 열(date) 제외
        T, _ = values.shape

        # 2) split
        tr_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        train_raw, val_raw, test_raw = values[:tr_end], values[tr_end:val_end], values[val_end:]

        # 3) 스케일러
        self.scaler = StandardScaler().fit(train_raw)
        self.train, self.val, self.test = (self.scaler.transform(x) for x in (train_raw, val_raw, test_raw))

        # 4) 레이블(z-score)
        self.train_labels = self._make_labels(train_raw, z_th)
        self.val_labels   = self._make_labels(val_raw,   z_th)
        self.test_labels  = self._make_labels(test_raw,  z_th)

        print(f"train: {self.train.shape}, val: {self.val.shape}, test: {self.test.shape}")

    @staticmethod
    def _make_labels(split_values: np.ndarray, z_th: float):
        """
        채널별 z-score → point-level 마스크(0/1) 생성 후,
        각 시점에 대해 **최소 한 채널이라도 이상치면 1** 로 축약한다.
        최종 shape : (T,)
        """
        z = zscore(split_values, axis=0, nan_policy="omit")             # (T, C)
        per_channel = (np.abs(z) > z_th)                                # (T, C)
        per_time    = per_channel.any(axis=1).astype(np.float32)        # (T,)
        return per_time

    # --- 필수 인터페이스 -------------------------
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Crypto'):
        dataset = CryptoSegLoader(csv_path=data_path + "/bitcoin_ethereum_transactions.csv",win_size=win_size,step=1,mode=mode)
    elif (dataset == 'bitcoin'):
        dataset = CryptoSegLoader(csv_path=data_path + "/bitcoin_transactions.csv",win_size=win_size,step=1,mode=mode)
    elif (dataset == 'ethereum'):
        dataset = CryptoSegLoader(csv_path=data_path + "/ethereum_transactions.csv",win_size=win_size,step=1,mode=mode)
    elif (dataset == 'bitcoin_log_diff'):
        dataset = CryptoSegLoader(csv_path=data_path + "/bitcoin_log_diff.csv",win_size=win_size,step=1,mode=mode)
    elif (dataset == 'ethereum_log_diff'):
        dataset = CryptoSegLoader(csv_path=data_path + "/ethereum_log_diff.csv",win_size=win_size,step=1,mode=mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
