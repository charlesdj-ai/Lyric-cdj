import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import copy
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import transforms
train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")

train_x = torch.tensor(train_x).T
train_x = train_x.to(torch.float32)
train_x = train_x.reshape(5000,28,28)
train_x = torch.unsqueeze(train_x, dim=1)
train_x_ori = train_x

print(train_x.shape)
train_y = torch.tensor(train_y).T
train_y = train_y.to(torch.float32)
train_y = torch.argmax(train_y, -1)
print(train_y.shape)
train_y_ori = train_y
train_set = Data.TensorDataset(train_x, train_y)



test_y = torch.tensor(test_y).T
test_y = test_y.to(torch.float32)
test_y = torch.argmax(test_y, -1)
test_x = torch.tensor(test_x).T
test_x = test_x.to(torch.float32)
test_x = test_x.reshape(5000,28,28)
test_x = torch.unsqueeze(test_x, dim=1)
test_set = Data.TensorDataset(test_x, test_y)
print(test_x.shape)
print(test_y.shape)

unlabeled_x=np.load("unlabeled_x.npy")
unlabeled_x=torch.tensor(unlabeled_x).T
unlabeled_x=unlabeled_x.to(torch.float32)
unlabeled_x=unlabeled_x.reshape(5000,28,28)

final_x=np.load("final_x.npy")
final_x=torch.tensor(final_x).T
final_x=final_x.to(torch.float32)
final_x=final_x.reshape(5000,28,28)
weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(0.286, 0.353)])
print(train_x.shape)
train_x = weak_transform(train_x)
#数据集装载
def train_data_process():
    global train_x
    
    train_x = weak_transform(train_x)
    train_loader = Data.DataLoader(dataset=train_set,  
                                   batch_size=32,  
                                   shuffle=True,  
                                   )
    print("The number of batch in train_loader:", len(train_loader))  

    return train_loader


test_loader = Data.DataLoader(dataset=test_set,  
                                   batch_size=32,  
                                   shuffle=True,  
                                   )
#测试过程
def test_data_process():
    test_data_x = test_x.type(torch.FloatTensor) / 255.0  
    test_data_x = torch.unsqueeze(test_x, dim=1)  
    test_data_y = test_y  
    print("test_data_x.shape:", test_x.shape)
    print("test_data_y.shape:", test_y.shape)
    return test_data_x, test_data_y
#无标签初始数据处理
def unlabeled_data_process():
    unlabeled_data_x = unlabeled_x.type(torch.FloatTensor) / 255.0  
    unlabeled_data_x = torch.unsqueeze(unlabeled_x, dim=1)  
    return unlabeled_data_x
#获取final_y
def final_x_process():
    final_data_x = final_x.type(torch.FloatTensor) / 255.0  
    final_data_x = torch.unsqueeze(final_x, dim=1)  
    return final_data_x
#网络结构
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()  
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,  
                                             out_channels=64,  
                                             kernel_size=3,  
                                             stride=1,  
                                             padding=1,  
                                             ),  
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),  
                                   nn.AvgPool2d(kernel_size=2,  
                                                stride=2,  
                                                ),  
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64,  
                                             out_channels=128,  
                                             kernel_size=3,  
                                             stride=1,  
                                             padding=0,  
                                             ),  
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),  
                                   nn.AvgPool2d(kernel_size=2,  
                                                stride=2,  
                                                ),  
                                   )
        self.classifier = nn.Sequential(nn.Linear(128*6*6, 2048),  
                                        nn.Dropout2d(0.5),
                                        nn.ReLU(),
                                        nn.Linear(2048,1024),
                                        nn.ReLU(),
                                        nn.Linear(1024,512),
                                        nn.Dropout2d(0.5),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),  
                                        nn.Dropout2d(0.5),
                                        nn.ReLU(),  
                                        nn.Linear(256,10),
                                        )

    
    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(x.size(0), -1)                       
        output = self.classifier(x)  
        return output

#训练过程
def train_model(model, traindataloader,testdataloader,criterion, optimizer, num_epochs=25): 
    best_model_wts = copy.deepcopy(model.state_dict())  
    
    best_acc = 0.0  
    train_loss_all = []  
    train_acc_all = []  
    val_loss_all = []  
    val_acc_all = []  
    since = time.time()  
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        train_loss = 0.0  
        train_corrects = 0  
        train_num = 0  
        val_loss = 0.0  
        val_corrects = 0  
        val_num = 0  
        
        for  step,(b_x, b_y) in enumerate(traindataloader):
             
                model.train()  
                output = model(b_x)  
                pre_lab = torch.argmax(output, 1)  
                loss = criterion(output, b_y)  
                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()  
                train_loss += loss.item() * b_x.size(0)  
                train_corrects += torch.sum(pre_lab == b_y.data)  
                train_num += b_x.size(0)  
        for step,(b_x,b_y) in enumerate(testdataloader):
                model.eval()  
                output = model(b_x)  
                pre_lab = torch.argmax(output, 1)  
                loss = criterion(output, b_y)  
                val_loss += loss.item() * b_x.size(0)  
                val_corrects += torch.sum(pre_lab == b_y.data)  
                val_num += b_x.size(0)  

        
        train_loss_all.append(train_loss / train_num)  
        train_acc_all.append(train_corrects.double().item() / train_num)  
        val_loss_all.append(val_loss / val_num)  
        val_acc_all.append(val_corrects.double().item() / val_num)  
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  
            best_model_wts = copy.deepcopy(model.state_dict())  
        time_use = time.time() - since  
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    model.load_state_dict(best_model_wts)  
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all}
                                 )  

    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    return model

#训练
def train_model_process(myconvnet,num):
    optimizer = torch.optim.Adam(myconvnet.parameters(), lr=0.0005,weight_decay=5e-4)  
    torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()  
    train_loader = train_data_process()  
    test_data_x, test_data_y = test_data_process() 
    myconvnet = train_model(myconvnet, train_loader,test_loader, criterion, optimizer, num_epochs=num)  

    
#伪标签处理
def train_model_unlabeled_process(myconvnet,div):
    
    global train_x,train_y
    unlabeled_data_x=unlabeled_data_process()
    myconvnet.eval()  
    output = myconvnet(unlabeled_data_x)  
    output=F.softmax(output,dim=1)
    pre_lab = output.data.max(1, keepdim=True)[1]  
    
    
    for j in range(output.shape[0]):
           if max(output[j])>=div:
            p = unlabeled_data_x[j]
            p=p.unsqueeze(0)
            train_x=torch.cat((train_x,p),0)
            train_y=torch.cat((train_y,pre_lab[j]),0)
            
def getfinal_y(myconvnet):
    final_x = final_x_process()
    myconvnet.eval()  
    output = myconvnet(final_x)  
    output=F.softmax(output,dim=1)
    pre_lab = output.data.max(1, keepdim=True)[1]  
    res = np.array(pre_lab).T
    np.save("final_y.npy",res)
if __name__ == '__main__':
    convnet = ConvNet()
    #反复训练优化伪标签
    train_model_process(convnet,8)
    train_model_unlabeled_process(convnet,0.8)
    train_set = Data.TensorDataset(train_x, train_y)
    train_model_process(convnet,8)
    train_x = train_x_ori
    train_y = train_y_ori
    train_model_unlabeled_process(convnet,0.9)
    train_set = Data.TensorDataset(train_x, train_y)
    train_model_process(convnet,8)
    train_x = train_x_ori
    train_y = train_y_ori
    train_model_unlabeled_process(convnet,0.95)
    train_set = Data.TensorDataset(train_x, train_y)
    train_model_process(convnet,20)
    getfinal_y(convnet)
    
    
