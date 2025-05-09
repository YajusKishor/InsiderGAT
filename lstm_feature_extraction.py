
from torch.nn import LSTM

import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from torch.nn.functional import one_hot
import torch.nn.functional as F
import os
from tool_and_model.early_stop_v1 import EarlyStopping
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import time

   
epochs= 1000

seq_len_list= [74] 


n_feat= 5 
lr=0.01
train_round=5


def process_data(data):

    data.fillna(0,inplace=True)
    x_columns= data.columns[4:4+seq_len]
    x = data[x_columns]
    x = np.array(x)
    x = torch.LongTensor(x)
    
    x = one_hot(x,num_classes=25)
    x = x.float()
    
    y = data['label'].to_list()
    y = torch.Tensor(y).int()
    return x, y


class LstmAutoEncoder(nn.Module):
    def __init__(self, input_size=25, hidden_size=100, batch_size=2000,encode_size=5):
        super(LstmAutoEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.encode_size = encode_size

        self.encoder_lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.encoder_fc = nn.Linear(self.hidden_size, self.encode_size)
        self.decoder_lstm = nn.LSTM(self.hidden_size, self.input_size, batch_first=True)
        self.decoder_fc = nn.Linear(self.encode_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.output_fc= nn.Linear(seq_len*self.encode_size, n_feat)
        self.decoder_input_fc=nn.Linear(n_feat, seq_len*self.encode_size)

    def forward(self, x):
        
        
        encoder_lstm, (n, c) = self.encoder_lstm(x)
        encoder_fc = self.encoder_fc(encoder_lstm)
        encoder_out= encoder_fc.flatten(1)
        encoder_out= self.output_fc(encoder_out)
        
        
        decoder_input= self.decoder_input_fc(encoder_out)
        decoder_input= decoder_input.view(-1,seq_len,n_feat)
        
        decoder_fc = self.decoder_fc(decoder_input)
        decoder_lstm, (n, c) = self.decoder_lstm(decoder_fc)
        return encoder_out,decoder_lstm.squeeze()


if __name__ == '__main__':

    sample_data_dir='./sample_data' 
    data= pd.read_csv(os.path.join(sample_data_dir,'1-data-test-combine-sequence.csv'))
    df_manual = pd.read_csv(os.path.join(sample_data_dir,'1-data-test-combine.csv'))
   
    result_dir= r'./lstm_feature_extraction_result'
    os.makedirs(result_dir,exist_ok=True)

    start= time.time()
    
    for seq_len in seq_len_list:
        
        
        perform= pd.DataFrame() 
        
        x, y = process_data(data)
        
        
        batch_data = Data.DataLoader(
            dataset=Data.TensorDataset(x, y), 
            batch_size=2000,  
            shuffle=True,  
        )

        
        for r in trange(train_round):
            
            train_acc_s = []
            train_loss_s = []
            test_acc_s = []
            test_loss_s = []
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = LstmAutoEncoder().to(device) 
            loss_function = nn.MSELoss()  
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_model_path = os.path.join(result_dir,'early_stop_model' )
            os.makedirs(best_model_path,exist_ok=True)
            best_model_path = os.path.join(best_model_path,f'check_seq{seq_len}_{r}round_best.pt')
            early_stopping = EarlyStopping(save_path=best_model_path,verbose=(True),patience=20,delta=0.0001,metric='loss')
        
            for epoch in range(epochs):
                model.train()
                for b_x, b_y in batch_data:
                    optimizer.zero_grad()
                    _, y_pred = model(b_x)  
                    loss = loss_function(y_pred, b_x)
                    loss.backward()
                    optimizer.step()
                
                train_loss_s.append(loss.item())
        
                
                model.eval()
                test_data = x[-500:]
                encode, y_pred = model(test_data)
                test_loss= loss_function(y_pred, test_data)
                test_loss_s.append(test_loss.item())
                
                early_stopping(test_loss,model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch:",epoch)
                    break

            early_stopping.draw_trend(train_loss_s, test_loss_s)
            dic= {'round':r,'min_loss':min(test_loss_s),'stop_epoch':epoch}
            perform= perform.append(dic,ignore_index=True)
            
        perform.to_csv(os.path.join(result_dir,f'check_perform_result_seq{seq_len}.csv'))

        r_min = perform['min_loss'].idxmin()
        
        model_extract = LstmAutoEncoder()  

        model_extract.load_state_dict(torch.load(best_model_path.replace(f'_{r}', f'_{r_min}')))
        print(best_model_path.replace(f'_{r}', f'_{r_min}'))
        model_extract.eval()
        output,_= model_extract(x)
        
        df_feature= pd.DataFrame(output.detach().numpy())
        df_save= pd.concat([df_manual,df_feature],axis=1)
        df_save.to_csv(os.path.join(result_dir,f'check_1-data-test-manual+lstm_seq{seq_len}.csv'),index=False)
    end= time.time()
    print('time used',end-start)

    

       


