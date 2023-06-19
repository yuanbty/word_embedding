import pandas as pd
import torch
from torch.utils.data import Dataset
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


class NewsDataset(Dataset):

    def __init__(self):
        '''load data'''
        self.df = pd.read_csv('/DIR_PATHWAY/small_sample2.csv')
        token_list = self.df_to_token(self.df)
        vocab, count, index, total = self.vocabulary(token_list)
        self.vocab = vocab
        self.count = count
        self.index = index
        training_data = self.get_train_data(token_list)

        self.data = torch.tensor(training_data, dtype = torch.int64)
    
    def __len__(self):
        '''Get the number of data in the Dataset'''
        return len(self.data)


    def __getitem__(self, id):
        '''Get an sample of data''' 
        center = self.data[id, 0]
        context = self.data[id, 1]
        return center, context
    
    def df_to_token(self, df):
        '''Convert df to list of tokens'''
        data_list = []
        text_list = df['Text'].values.tolist()
        for text in text_list:
            pattern = r'[0-9]'
            new_s = re.sub(pattern, '', text)
            
            rem_punc_tok = RegexpTokenizer(r'\w+')
            tokens = rem_punc_tok.tokenize(new_s)

            words = [word.lower() for word in tokens]

            stop_words = set(stopwords.words('english'))

            words = [word for word in words if not word in stop_words]

            data_list.append(words)

        return data_list


    def get_train_data(self, token_list):
        '''Convert tokens to trainable pair'''
        context_size = 2
            
        vocab_index = list(range(len(self.vocab)))
        training_data = []
        for text in token_list:
            indices = []
            for word in text:
                indices.append(self.index[word])

            for center_pos in range(len(indices)):

                for d in range(-context_size, context_size+1):
                    context_pos = center_pos + d

                    if context_pos < 0 or context_pos >= len(indices) or center_pos == context_pos:
                        continue
                        
                    center_id = indices[center_pos]
                    context_id = indices[context_pos]

                    if center_id == context_id:
                        continue
                        
                    training_data.append([center_id, context_id])

        return training_data



    def vocabulary(self,list_of_list):
        '''Build a vocabulary inventory'''

        vocab_dict = {}
        count_dict = {}
        index_dict = {}
        total = 0

        for text in list_of_list:
            for word in text:
                if word not in count_dict:
                    count_dict[word] = 0
                    vocab_dict[len(vocab_dict)] = word
                    index_dict[word] = len(vocab_dict)
                count_dict[word] += 1
                total += 1

        return vocab_dict, count_dict, index_dict ,total

    def minimize_data(self, vocab, count, index):
        """Aim to delelte work with certain amount of occurance
        NOT IMPLEMENTED
        """
        
        return



        
