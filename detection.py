import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,accuracy_score
import csv
from torch_geometric.data import Data
from tool_and_model.early_stop_v1 import EarlyStopping
from tqdm import trange
from tool_and_model.model import WorkingModel


LR= 0.01
train_round=10
dic_draw={}
EPOCH= 1000


f1_L= ['manual'] 

f2_L= ['lstm_seq74']

feat_list=[] 
for f1 in f1_L:
    for f2 in f2_L:
        feat_list.append(f1 + '+' + f2)
CNN='CNN'  
GNN='GAT'  
Residual= 'YES' 

file_dir= './sample_data'
all_result= './detection_result'
result_dir= os.path.join(all_result,'GAT_results') 
os.makedirs(result_dir,exist_ok=True)

df_e= pd.read_csv(os.path.join(file_dir,'1-data-test-undirected_edge.csv'))
edge= df_e.to_numpy().T
edge_index = torch.from_numpy(edge)

for feat in  feat_list:
    
    perform_file=os.path.join(result_dir,f'GAT_result.csv')
    with open(perform_file, 'w', newline='') as f:
        writer = csv.writer(f)
        my_list = ['Features', 'round','test_acc','test_pre','test_rec','test_f1','epoch']
        writer.writerow(my_list)
    plot_file=perform_file.replace('results.csv','plot.npy')
    
    df_data= pd.read_csv(os.path.join(file_dir,'1-data-test-'+ feat + '.csv'))
    
    cols= df_data.columns[5:]
    
    x = df_data[cols].to_numpy(dtype=np.float32)
    x = torch.from_numpy(x)  
    
    y = df_data['label'].to_numpy()
    y = torch.from_numpy(y)                               
    
    data = Data(x=x,
                edge_index=edge_index,
                y=y)
    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.train_mask[:data.num_nodes - 572] = 1                  
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
    data.test_mask[data.num_nodes - 572:] = 1
    
    for r in trange(train_round):
        
        train_loss_s = []
        train_acc_s = []
        train_rec_s = []
        train_pre_s = []
        train_f1_s = []
    
        test_loss_s = []
        test_acc_s = []
        test_rec_s = []
        test_pre_s = []
        test_f1_s = []
    
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        model = WorkingModel(input_dim=data.num_node_features, output_dim=data.num_node_features,residual=Residual).to(device)         
        data = data.to(device)                                                       
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    
        best_model_path = os.path.join(result_dir,"early_stop_model" )
        os.makedirs(best_model_path,exist_ok=True)
        best_model_path = os.path.join(best_model_path,f'{CNN}_{GNN}_{feat}_{Residual}_{r}round_best.pt')
        early_stopping = EarlyStopping(save_path=best_model_path,verbose=(True),patience=30,delta=0.0001,metric='loss')
        for epoch in range(EPOCH):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            _, out1 = out.max(dim=1)
            pred_y = torch.masked_select(out1, data.train_mask.bool()).tolist()
            true_y = data.y[data.train_mask.bool()].tolist()
            train_acc_s.append(accuracy_score(true_y, pred_y))
            loss = F.nll_loss(F.log_softmax(out[data.train_mask.bool()], dim=1), data.y[data.train_mask.bool()].long())
            loss.backward()
            optimizer.step()
            train_loss_s.append(loss.item())
           
            model.eval()
            out = model(data)
            _, out1 = out.max(dim=1)
            pred_y = torch.masked_select(out1, data.test_mask.bool()).tolist()
            true_y = data.y[data.test_mask.bool()].tolist()
            test_acc = accuracy_score(true_y, pred_y)
            test_acc_s.append(test_acc)
            test_loss = (F.nll_loss(F.log_softmax(out[data.test_mask.bool()],dim=1), data.y[data.test_mask.bool()].long()))
            test_loss_s.append(test_loss.item())
            early_stopping(test_loss,model)
            if early_stopping.early_stop:
                print("Early stopping at epoch:",epoch)
                break
        
        early_stopping.draw_trend(train_loss_s, test_loss_s)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        out = model(data)
        _, out1 = out.max(dim=1)
        pred_y = torch.masked_select(out1, data.test_mask.bool()).tolist()
        true_y = data.y[data.test_mask.bool()].tolist()
        test_acc = accuracy_score(true_y, pred_y)
        test_pre = precision_score(true_y, pred_y)
        test_rec = recall_score(true_y, pred_y)
        test_f1 = f1_score(true_y, pred_y)
        print(f'best model testing performance for round {r}:')
        print(f'Accuracy\tPrecision\tRecall\tF1')
        print(f'{test_acc:.2f}\t\t{test_pre:.2f}\t\t{test_rec:.2f}\t{test_f1:.2f}')
        
        feature= feat
    
        with open(perform_file, 'a', newline='') as f:
            writer = csv.writer(f)
            my_list = [feature, r , test_acc, test_pre, test_rec, test_f1,epoch]
            writer.writerow(my_list)
        print('finished round:',r)
    
    # load best model

