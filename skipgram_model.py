import torch.nn as nn
import numpy as np
import torch.nn.functional as funl
import torch


class Skipgram(nn.Module):

    def __init__(self, vocab_size):
        super(Skipgram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_center = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = 300,
            max_norm = 1
            )

        self.embedding_context = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = 300,
            max_norm = 1
            )
        """self.linear_center = nn.Linear(
            in_features = 300,
            out_features = self.vocab_size
        )

        self.linear_context = nn.Linear(
            in_features = 300,
            out_features = self.vocab_size
        )""" #Deleted for less computational cost

        """If use crossentropyLoss"""
        #self.loss = nn.CrossEntropyLoss()
        

    def forward(self, center_input, context_input):
        #print('center_input.shape: ', center_input.shape)
        #print('center_input.shape: ', context_input.shape)
        
        x_center = self.embedding_center(center_input)
        x_context = self.embedding_context(context_input)
        #except IndexError:
            #x_center = self.embedding_center
            #x_context = self.embedding_context
        #print('x_conter.shape: ', x_center.shape)
        
        #print('x_context.shape: ', x_context.shape)
        
        #x_center = self.linear_center(x_center)
        #x_context = self.linear_context(x_context)

        product = torch.mul(x_center, x_context)
        add = torch.sum(product, dim = 1)
        loss = funl.logsigmoid(add)

        return -loss.mean()
