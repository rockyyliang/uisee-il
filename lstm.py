import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch
import torchvision

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes,kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)),0.1)
        return out

class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x

class Baseline1D_LSTM(nn.Module):
    def __init__(self,latent_dim = 128,lstm_layers=3,hidden_dim=64):
        super(Baseline2D_LSTM, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained = True)
        # self.linear = nn.Linear(1000, 128)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.lstm_s = LSTM(latent_dim, lstm_layers, hidden_dim)
        # self.lstm_a = LSTM(latent_dim, lstm_layers, hidden_dim)
        # self.fc1_a = nn.Linear(64,100)
        # self.fc2_a = nn.Linear(100, 10)
        # self.fc3_a = nn.Linear(10,1)
        # self.fc1_s = nn.Linear(64,100)
        # self.fc2_s = nn.Linear(100, 10)
        # self.fc3_s = nn.Linear(10,1)
        # self.drop = nn.Dropout(0.2)
        self.block1 = BasicBlock(3, 24, 7, 2)
        self.block2 = BasicBlock(24, 48, 7, 2)
        self.block3 = BasicBlock(48, 64, 5, 2)
        self.block4 = BasicBlock(64, 96, 5,2)
        self.block5 = BasicBlock(96, 128, 5,2)
        self.block6 = BasicBlock(128, 128, 5,2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        #self.lstm_s = LSTM(latent_dim, lstm_layers, hidden_dim)

        self.lstm_a = LSTM(latent_dim, lstm_layers, hidden_dim)
        self.fc1_a = nn.Linear(64,100)
        self.fc2_a = nn.Linear(100, 10)
        self.fc3_a = nn.Linear(10,1)

        #self.fc1_s = nn.Linear(64,100)
        #self.fc2_s = nn.Linear(100, 10)
        #self.fc3_s = nn.Linear(10,1)

        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)

        # x = self.resnet(x)
        # x = self.linear(x)
        # x = self.pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x)
        x = x.view(batch_size, seq_length, -1)

        a = self.lstm_a(x)

        #s = self.lstm_s(x)

        a = a[:,-1,:]

        #s = s[:,-1,:]

        a = self.drop(F.relu(self.fc1_a(a)))
        a = self.drop(F.relu(self.fc2_a(a)))
        a = self.fc3_a(a)

        #s = self.drop(F.relu(self.fc1_s(s)))
        #s = self.drop(F.relu(self.fc2_s(s)))
        #s = self.fc3_s(s)

        return a.view(-1) #,s.view(-1)

class Baseline2D_LSTM(nn.Module):
    def __init__(self,latent_dim = 192,lstm_layers=3,hidden_dim=64):
        super(Baseline2D_LSTM, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained = True)
        # self.linear = nn.Linear(1000, 128)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.lstm_s = LSTM(latent_dim, lstm_layers, hidden_dim)
        # self.lstm_a = LSTM(latent_dim, lstm_layers, hidden_dim)
        # self.fc1_a = nn.Linear(64,100)
        # self.fc2_a = nn.Linear(100, 10)
        # self.fc3_a = nn.Linear(10,1)
        # self.fc1_s = nn.Linear(64,100)
        # self.fc2_s = nn.Linear(100, 10)
        # self.fc3_s = nn.Linear(10,1)
        # self.drop = nn.Dropout(0.2)
        self.block1 = BasicBlock(3, 24, 7, 2)
        self.block2 = BasicBlock(24, 48, 7, 2)
        self.block3 = BasicBlock(48, 64, 5, 2)
        self.block4 = BasicBlock(64, 96, 5,2)
        self.block5 = BasicBlock(96, 128, 5,2)
        self.block6 = BasicBlock(128, 128, 5,2)

        self.speed_linear1 = nn.Linear(1, 8)
        self.speed_linear2 = nn.Linear(8, 64)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm_s = LSTM(latent_dim, lstm_layers, hidden_dim)
        self.lstm_a = LSTM(latent_dim, lstm_layers, hidden_dim)
        self.fc1_a = nn.Linear(64,100)
        self.fc2_a = nn.Linear(100, 10)
        self.fc3_a = nn.Linear(10,1)
        self.fc1_s = nn.Linear(64,100)
        self.fc2_s = nn.Linear(100, 10)
        self.fc3_s = nn.Linear(10, 1)
        self.drop = nn.Dropout(0.2)
    def forward(self, x, speed):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        speed = speed.view(batch_size * seq_length, 1)
        # x = self.resnet(x)
        # x = self.linear(x)
        # x = self.pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        # measurement
        speed = F.relu(self.speed_linear1(speed))
        speed = F.relu(self.speed_linear2(speed))

        x = self.pool(x)
        x = x.view(batch_size * seq_length, -1)
        # print(x.shape)
        # print(speed.shape)
        x = torch.cat([x, speed], dim=1)
        x = x.view(batch_size, seq_length, -1)

        a = self.lstm_a(x)
        s = self.lstm_s(x)
        a = a[:,-1,:]
        s = s[:,-1,:]
        a = self.drop(F.relu(self.fc1_a(a)))
        a = self.drop(F.relu(self.fc2_a(a)))
        a = self.fc3_a(a)
        s = self.drop(F.relu(self.fc1_s(s)))
        s = self.drop(F.relu(self.fc2_s(s)))
        s = self.fc3_s(s)
        s = torch.sigmoid(s)
        return a.view(-1), s
# net = Baseline2D_LSTM()
# input = torch.randn(3,10,3,360,640)
# angle,speed = net(input)
# print(angle)
