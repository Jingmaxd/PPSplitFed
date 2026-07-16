#PPSplitFed-MLP-2:784 → 256 →128|split|64 → 10 
import sys
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pandas import DataFrame
import matplotlib.pyplot as plt
import copy
from torchvision import datasets
import logging
import numpy as np
import random
from crypto_all import execute_ndarray,request_key_ndarray,serverexecute_ndarray,m_execute,m_request_key,m_serverexecute

logger = logging.getLogger(__name__)

#mife
y_prime = np.array(
    [
        [0.2],
        [0.2],
        [0.2],
        [0.2],
        [0.2]
    ])

enrolled_parties = {
    'id_1': 1,
    'id_2': 1,
    'id_3': 1,
    'id_4': 1,
    'id_5': 1
}
    
    
#－－－－－－－－－SFL training－－－－－－－－－－－－
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

program = "PPSFL-MLP2"
print(f"---------{program}----------")         

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

num_users = 5
epochs = 20
frac = 1     
lr = 0.1

#client model
class Client_side(nn.Module):
    def __init__(self):
        super(Client_side, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256,128)
               
    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net_glob_client = Client_side()
net_glob_client.to(device)
print('net_glob_client:', net_glob_client)     

#server model
class Server_side0(nn.Module):
    def __init__(self):
        super(Server_side0, self).__init__()
        self.fc3 = nn.Linear(128,64)
                   
    def forward(self, x):
        x = F.relu(self.fc3(x))
        return x

class Server_side1(nn.Module):
    def __init__(self):
        super(Server_side1, self).__init__()
        self.fc4 = nn.Linear(64, 10)
               
    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net_glob_server0 = Server_side0() 
net_glob_server0.to(device)
   
net_glob_server1 = Server_side1() 
net_glob_server1.to(device)
  
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []
criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_secure(w,y):
    ct = dict()
    ct['parties'] = enrolled_parties
    ct['ct_dict'] = dict()
    
    sk = m_request_key(y, {'parties': enrolled_parties})
    
    w_avg = copy.deepcopy(w[0])
    num=0
    for k in w_avg.keys():
        num+=1
        ct['ct_dict']['id_1'] = m_execute(np.array(w[0][k]), {'id': 'id_1'})
        ct['ct_dict']['id_2'] = m_execute(np.array(w[1][k]), {'id': 'id_2'})
        ct['ct_dict']['id_3'] = m_execute(np.array(w[2][k]), {'id': 'id_3'})
        ct['ct_dict']['id_4'] = m_execute(np.array(w[3][k]), {'id': 'id_4'})
        ct['ct_dict']['id_5'] = m_execute(np.array(w[4][k]), {'id': 'id_5'})
        
        dec= m_serverexecute(sk, ct, y,{'type': 'mife'})
        w_avg[k]=torch.tensor(dec)
        if num  == 1:
            w_avg[k]=w_avg[k].reshape(256, 784)
        elif num == 3:
            w_avg[k]=w_avg[k].reshape(128, 256)
        
    return w_avg

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

w_glob_server0 = net_glob_server0.state_dict()
w_glob_server1 = net_glob_server1.state_dict()

w_locals_server0 = []
w_locals_server1 = []

#client idx collector
idx_collect = []
l_epoch_check = False
fed_check = False

net_model_server0 = [net_glob_server0 for i in range(num_users)]
net_server0 = copy.deepcopy(net_model_server0[0]).to(device)

net_model_server1 = [net_glob_server1 for i in range(num_users)]
net_server1 = copy.deepcopy(net_model_server1[0]).to(device)

def grad_relu(x):
    x[x<0]=0
    x[x>0]=1   
    return x

def _encode_labels( y, k):
    onehot = np.zeros((y.shape[0],k))
    for idx, val in enumerate(y):
        onehot[idx,val] = 1.0
    return onehot

  
#server-side training  
def train_server(fx_ct,fx_ct_T, y, onehot_y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server0,net_model_server1, criterion, optimizer_server0,optimizer_server1, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server0,w_locals_server1,net_server0, net_server1
    global loss_train_collect_user, acc_train_collect_user, lr
    
    
    net_server0 = copy.deepcopy(net_model_server0[idx]).to(device)
    net_server0.train()
    optimizer_server0 = torch.optim.SGD(net_server0.parameters(), lr = lr)
    optimizer_server0.zero_grad()
    net_server1 = copy.deepcopy(net_model_server1[idx]).to(device)
    net_server1.train()
    optimizer_server1 = torch.optim.SGD(net_server1.parameters(), lr = lr)
    optimizer_server1.zero_grad()

    #print('------------forward propagation------')
    fc3_b=net_server0.fc3.bias.data 
    fc3_w=net_server0.fc3.weight.data 
    fc3_wn=fc3_w.numpy()
    fc3_bn=fc3_b.numpy()

    sk_w0 = request_key_ndarray(fc3_wn)
    z1= serverexecute_ndarray(sk_w0, fx_ct, fc3_wn)
    z1=torch.Tensor(np.transpose(z1))
    z1= z1+fc3_b
    a1=F.relu(z1)
    a1 =a1.clone().detach().requires_grad_(True)
    a1 = a1.to(device)
    y = y.to(device)
    a2 = net_server1(a1)

    #compute loss and acc
    loss = criterion(a2, y)
    acc = calculate_accuracy(a2, y)

    #print('------------backward propagation----------') 
    loss.backward()   
    da1 = a1.grad.clone().detach()   
    da1=da1.numpy()
    dz1=da1.T * grad_relu(z1.numpy().T)

    sk_sigma0 = request_key_ndarray(dz1)
    grad_w1= serverexecute_ndarray(sk_sigma0, fx_ct_T, dz1)     
    dw1= grad_w1
    b1_ones=np.ones(60)
    db1= dz1.dot (b1_ones)

    dfx= dz1.T.dot(fc3_wn)
    dfx=torch.Tensor(dfx)

    #update model
    optimizer_server1.step() 
    w3_new =  fc3_wn - lr * dw1 
    b3_new =  fc3_bn - lr * db1 
    w3_new=torch.Tensor(w3_new)
    b3_new=torch.Tensor(b3_new)   
    model_dict={'fc3.weight': w3_new, 'fc3.bias': b3_new}
    net_server0.load_state_dict(model_dict)
    
    #print('------------backward propagation ends------')
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server1[idx] = copy.deepcopy(net_server1)
    net_model_server0[idx] = copy.deepcopy(net_server0)
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)    
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        w_server0 = net_server0.state_dict() 
        w_server1 = net_server1.state_dict()
                    
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server0.append(copy.deepcopy(w_server0))
            w_locals_server1.append(copy.deepcopy(w_server1))
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display  
            w_glob_server0 = FedAvg(w_locals_server0)   
            w_glob_server1 = FedAvg(w_locals_server1) 
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server0.load_state_dict(w_glob_server0)    
            net_model_server0 = [net_glob_server0 for i in range(num_users)]
            
            net_glob_server1.load_state_dict(w_glob_server1)    
            net_model_server1 = [net_glob_server1 for i in range(num_users)]
            
            w_locals_server0 = []
            w_locals_server1 = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server0, net_model_server1,criterion, batch_acc_test, batch_loss_test, check_fed, net_server, net_glob_server 
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, w_glob_server, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    
    net0 = copy.deepcopy(net_model_server0[idx]).to(device)
    net1 = copy.deepcopy(net_model_server1[idx]).to(device)
    net0.eval()
    net1.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward propagation-------------
        a=net0(fx_client)
        fx_server = net1(a)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
               
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                
            # if federation is happened----------                    
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("------ Federation process at Server-Side ------- ")
                print("------------------------------------------------")
                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")
         
    return 

#Clients-side Program
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 60, shuffle = True,drop_last=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 60, shuffle = True,drop_last=True)
        
    #client model training
    def train(self, net):
        global list_ts,list_tc
        net.train()
        optimizer_client = torch.optim.SGD(net.parameters(), lr = self.lr) 
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
        
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #print('---------client side forward propagation------------------')
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                fx_array = client_fx.detach().numpy()

                #activations encryption
                ct_ff_lst = execute_ndarray(fx_array)
                ct_bf_lst = execute_ndarray(fx_array.T)

                #print('---------server side forward+backward propagation------------------')                    
                onehot_labels= _encode_labels( labels, 10)
                dfx = train_server(ct_ff_lst, ct_bf_lst,labels, onehot_labels,iter, self.local_ep, self.idx, len_batch)                           
        
                #print('---------client side backward propagation-----------------')
                fx.backward(dfx)
                optimizer_client.step()

        return net.state_dict() 
    

    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
               
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
                        
        return          


# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client 
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                      
BATCH_SIZE = 60
dataset_train = datasets.MNIST(root='data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)
dataset_test = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())
train_iterator = DataLoader(dataset=dataset_train, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,drop_last=True)
test_iterator = DataLoader(dataset=dataset_test, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False,drop_last=True)
  
print(f'Number of training examples: {len(train_iterator)}')
print(f'Number of testing examples: {len(test_iterator)}')

for x, y in train_iterator:
    print("shape of x = ", x.shape)
    print(type(x))
    break

dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

#------------ Training And Testing  -----------------
net_glob_client.train()
w_glob_client = net_glob_client.state_dict()

for iter in range(epochs):
    print('iter:',iter)
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    w_locals_client = []
    
    ct = dict()
    ct['parties'] = enrolled_parties
    ct['ct_dict'] = dict()
    for idx in idxs_users:
        print('user idx :',idx )
        local = Client(net_glob_client, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        # Testing -------------------
        local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
               
    #print("------  Federation process of Client-Side Model------- ")
    
    w_glob_client = FedAvg_secure(w_locals_client,y_prime)   
    # Update client-side global model 
    net_glob_client.load_state_dict(w_glob_client)    
      
print("Training and Evaluation completed!")    

# Save output data to .excel file (we use for comparision plots)
round_process = [i for i in range(1, len(acc_train_collect)+1)]
df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
file_name = program+".xlsx"    
df.to_excel(file_name, sheet_name= "v1_test", index = False)     
