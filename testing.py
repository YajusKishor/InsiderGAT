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


f1_L= ['manual'] 

f2_L= ['lstm_seq74']

feat_list=[] 
for f1 in f1_L:
    for f2 in f2_L:
        feat_list.append(f1 + '+' + f2)

file_dir= './sample_data'


df_e= pd.read_csv(os.path.join(file_dir,'1-data-test-undirected_edge.csv'))
edge= df_e.to_numpy().T
edge_index = torch.from_numpy(edge)

for feat in feat_list:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorkingModel(input_dim=data.num_node_features, output_dim=data.num_node_features,residual='YES').to(device)         
    data = data.to(device)                                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)