#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

import matplotlib as plt
import torch

from torch.optim import lr_scheduler
import torchvision
from opt import opt
from data import Data
from network import REID_NET
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc

def evaluate(model, query_loader,test_loader,queryset,testset):
    model.eval()

    print('extract features, this may take a few minutes')
    qf = extract_feature(model, tqdm(query_loader)).numpy()
    gf = extract_feature(model, tqdm(test_loader)).numpy()

    def rank(dist):
        r = cmc(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
        m_ap = mean_ap(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras)

        return r, m_ap


        ######################### evaluation without rank##########################
    dist = cdist(qf, gf)

    r, m_ap = rank(dist)

    print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

if __name__ == '__main__':
    data = Data()
    model = REID_NET()
    # print(model)
    model = model.to('cuda')
    model.load_state_dict(torch.load(opt.weight))
    model.eval()
    evaluate(model,data.query_loader,data.test_loader,data.queryset,data.testset)