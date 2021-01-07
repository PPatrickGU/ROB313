import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torch.nn.functional as F
from opt import opt

num_classes = 751  # change this depend on your dataset


class REID_NET(nn.Module):
    def __init__(self):
        # write the CNN initialization
        super(REID_NET, self).__init__()
        model = resnet50(pretrained=True)
        self.fc_hidden1=1024
        self.fc_hidden2=768
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1])) # delete the last FC layer of resnet

        fc_features = model.fc.in_features

        self.fc_id = nn.Sequential(
                        nn.Linear(fc_features, self.fc_hidden1),
                        nn.Linear(self.fc_hidden1, num_classes)
                        )
        self.fc_metric = nn.Sequential(
                        nn.Linear(fc_features, self.fc_hidden1),
                        nn.Linear(self.fc_hidden1, num_classes)
                        )

    def forward(self, x):
        # write the CNN forward
        print('begin training')

        x = self.resnet(x).squeeze()  # ResNet without the last FC layer

        predict_id= self.fc_id(x) # Write this layer

        predict_metric= self.fc_metric(x) # Write this layer

        predict = torch.cat([predict_id, predict_metric], dim=1)
        # print(predict.shape)

        return predict, predict_id, predict_metric
