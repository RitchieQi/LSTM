import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self,ninp,nhid,nlayers,nclass,dropout = 0.5):
        super(LSTM,self).__init__()
        ''' input size = 17(features)*2000(steps)'''
        self.lstm = nn.LSTM(input_size = ninp,hidden_size = nhid, num_layers = nlayers, dropout = dropout, batch_first = True)
        self.fc1 = nn.Linear(nhid,512)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128)
        self.drop2 = nn.Dropout(0.4)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,nclass)
    def forward(self,x,hidden=None):
        x,hidden = self.lstm(x,hidden)
        x = x[:,-1,:]
        
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss,self).__init__()
    def forward(self,pred,label):
        loss = F.cross_entropy(pred, label)
        #loss = F.nll_loss(pred, label)
        #loss = nn.CrossEntropyLoss(pred,label)
        return loss


