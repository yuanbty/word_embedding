import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import math

class Trainer:

    def __init__(self, model, epochs, dataloader,batch_size ,optimizer, device, model_dir ,train_dataset):
        self.model = model
        self.epochs = epochs
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir
        self.data = train_dataset.data
        self.count = train_dataset.count
        self.index = train_dataset.index
        self.losses = []
        self.model.to(self.device)

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            print('\n===== EPOCH {}/{} ====='.format(epoch + 1, self.epochs)) 

            #print([item[0] for item in list(enumerate(self.dataloader))])
            for i, (center_batch, context_batch) in enumerate(self.dataloader):
                
                #print(i)
                #if i == (math.floor(len(self.data)/self.batch_size)-1):
                if i == 24402:
                    break
                self.model.train()

                center_batch = center_batch.to(self.device)
                #print(center_batch.shape)
                context_batch = context_batch.to(self.device)

                self.optimizer.zero_grad()

                loss = self.model(center_batch, context_batch)
                loss.backward()

                self.optimizer.step()

                self.losses.append(loss.item())
                
                

                if i % 200 == 0:
                    print(f'Batch: {i+1}/{len(self.dataloader)}, Loss: {loss.item()}')    
                    

            torch.save({'model_state_dict': self.model.state_dict(), 
                    'losses': self.losses,
                    'count': self.count,
                    'index': self.index
                    },                  
                    '{}/model{}.pth'.format(self.model_dir, epoch))




    


            
            


   