import numpy as np
import pandas as pd
import torch
import sys
from skipgram_model import Skipgram
from dataset import NewsDataset
from sklearn.manifold import TSNE



train_dataset = NewsDataset()
vocab_size = len(train_dataset.vocab)
model = Skipgram(vocab_size)
model.load_state_dict(torch.load('MODELNAME.pth')['model_state_dict'])
embedding_layer = model.embedding_center.weight.data

loss = torch.load('MODELNAME.pth')['losses']
count = torch.load('MODELNAME.pth')['count']
index_word = torch.load('MODELNAME.pth')['index']
vocab = train_dataset.vocab


def visualize(embedding):
    '''Visualize Embedding'''
    tsne = TSNE(n_components = 2).fit_transform(embedding_layer.cpu())
    x, y = [], []
    annotations = []
    for idx, coord in enumerate(tsne):
    
        annotations.append(vocab[idx])
        x.append(coord[0])
        y.append(coord[1]) 

    #Only draw words that has over 50 occurrences
    plot_words = [word for word, occurrences in count.items() if occurrences >= 50]
    plot_words_len = len(plot_words)

    plt.figure(figsize = (12.8, 9.6), dpi = 180)
    for word in plot_words:
    
    vocab_idx = index_word[word]
        
    plt.scatter(x[vocab_idx], y[vocab_idx], s = 3)
    plt.annotate(word, xy = (x[vocab_idx], y[vocab_idx]), ha='right',va='bottom', fontsize=5)
    #plt.savefig('visual50.png')   <--- save your figure
    plt.show()
    return x, y

def most_similar(word, topN, coord_x, coord_y):
    '''Find the most similar words to the input word and return a dictionary of result'''
    if word not in vocab.values():
        print('Word not in vocabury!')
        return 
    else:
        word_id = list(vocab.keys())[list(vocab.values()).index(word)] 
        x_word = coord_x[word_id]
        y_word = coord_y[word_id]
        result = {}
        index = 0
        for x, y in zip(coord_x, coord_y):
            eu = np.sqrt((x_word - x)**2 + (y_word - y)**2)
            result[vocab[index]] = eu
            index += 1
        
        result = sorted(result.items(), key=lambda x:x[1])[1:topN+1]
        return result
   
x, y = visualize(embedding_layer)
TOPN = 15
WORDYOUWANT = 'bank'
print('The Top',TOPN, 'similar words of', WORDYOUWANT, 'are')
print(most_similar(WORDYOUWANT, TOPN, x, y))
