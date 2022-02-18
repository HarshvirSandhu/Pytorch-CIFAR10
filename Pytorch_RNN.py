import pandas as pd
import torch
import torchtext
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

seed = 12
torch.manual_seed(seed)
random.seed(seed)
vocab_size = 10000
learning_rate = 3e-4
batch_size = 128
num_epochs = 15

embedding_dim = 128
hidden_dim = 256
num_classes = 2

df = pd.read_csv("C:/Users/harsh/Downloads/fake-news/train.csv")
# print(df.head(), df.columns)
del df['id']
del df['title']
del df['author']
if os.path.isfile('train.csv') is False:
    df.to_csv("train.csv", index=None)
del df

TEXT = torchtext.legacy.data.Field(tokenize='spacy',
                                   tokenizer_language='en_core_web_sm')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

fields = [('text', TEXT), ('label', LABEL)]
dataset = torchtext.legacy.data.TabularDataset('train.csv', format='csv',
                                               fields=fields, skip_header=True)
train, test = dataset.split(split_ratio=[0.75, 0.25])
# print(vars(train.examples[0]))

TEXT.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)
print(TEXT.vocab.freqs.most_common(20))
print(LABEL.vocab.stoi)
train_loader, test_loader = torchtext.legacy.data.BucketIterator.splits((train, test), batch_size=batch_size,
                                                                        sort_within_batch=False,
                                                                        sort_key=lambda x: len(x.TEXT))


class Classifier(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dimension, output_dim):
        super(Classifier, self).__init__()
        self.embed_layer = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dimension)
        self.fc = nn.Linear(hidden_dimension, output_dim)

    def forward(self, text):
        embed = self.embed_layer(text)
        output, (hidden, cell) = self.rnn(embed)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return F.softmax(output)


model = Classifier(input_dim=len(TEXT.vocab), embed_dim=embedding_dim,
                   hidden_dimension=hidden_dim, output_dim=num_classes)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    print(epoch)
    for num, batch in enumerate(train_loader):
        print(num/len(train_loader))
        text = batch.text
        label = batch.label
        print(text, label)
        score = model(text)
        loss = criterion(score, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
