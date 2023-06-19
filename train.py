import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skipgram_model import Skipgram
from dataset import NewsDataset
from trainer import Trainer


train_dataset = NewsDataset()
loader = DataLoader(train_dataset, batch_size = 96, shuffle = False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
vocab_size = len(train_dataset.vocab)
model = Skipgram(vocab_size).to(device)

epochs = 3
optimizer = optim.Adam(model.parameters(), lr = 0.025)


batch_size = 96
model_dir = '.'
trainer = Trainer(model, epochs,loader, batch_size ,optimizer, device, model_dir, train_dataset)

trainer.train()