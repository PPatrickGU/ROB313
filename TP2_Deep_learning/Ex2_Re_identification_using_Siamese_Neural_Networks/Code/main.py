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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model,train_loader,scheduler,optimizer,loss_function):
    # PLEASE COMPLETE THE TRAINING FUNCTION
    model.train()
    for (inputs, targets) in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        # zeroes the gradient buffers of all parameters
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()

def evaluate(model, query_loader,test_loader,queryset,testset):
    model.eval()

    print('extract features, this may take a few minutes')
    qf = extract_feature(model, tqdm(query_loader)).numpy() # 34*1502
    gf = extract_feature(model, tqdm(test_loader)).numpy() # 46*1502

    def rank(dist):
        r = cmc(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
        m_ap = mean_ap(dist, queryset.ids, testset.ids, queryset.cameras, testset.cameras)

        return r, m_ap
        ######################### evaluation without rank##########################
    dist = cdist(qf, gf) # 34*46

    r, m_ap = rank(dist)

    print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))



if __name__ == '__main__':


    data = Data()


    model = REID_NET()
    # print(model)
    model = model.to('cuda')
    loss_function = Loss()
    optimizer = get_optimizer(model)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    if opt.mode == 'train':
        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            # train(model,data.train_loader,scheduler,optimizer,loss_function)
            if epoch % 1 == 0:
                print('\nstart evaluate')
                evaluate(model,data.query_loader,data.test_loader,data.queryset,data.testset)
                os.makedirs('weights', exist_ok=True)
                torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        evaluate(model,data.query_loader,data.test_loader,data.queryset,data.testset)
