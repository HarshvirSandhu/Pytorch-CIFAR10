import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy.data import BucketIterator
import torch.optim as optim

TEXT = torchtext.legacy.data.Field(tokenize='spacy',
                                   tokenizer_language='en_core_web_sm')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

fields = [('label', LABEL), ('text', TEXT)]
dataset = torchtext.legacy.data.TabularDataset(path='C:/Users/harsh/Downloads/SpamClassifier-master/SpamClassifier'
                                                    '-master/smsspamcollection/SMSSpamCollection', format='tsv',
                                               fields=fields, skip_header=True)
TEXT.build_vocab(dataset, max_size=8000)
LABEL.build_vocab(dataset)

train, test = dataset.split([0.75, 0.25])
train_iterator, test_iterator = BucketIterator.splits((train, test), batch_size=32, sort_key=lambda x: len(x.TEXT),
                                                      sort_within_batch=False, shuffle=True)
embedding_dim = 10000


class Classifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Classifier, self).__init__()
        self.embed = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embed(x)
        out, (hidden, cell) = self.lstm(x)
        hidden.squeeze_(0)
        out = self.fc(hidden)
        return F.softmax(out)


model = Classifier(input_dim=len(TEXT.vocab), embedding_dim=embedding_dim, hidden_dim=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
num_epochs = 10

for epoch in range(num_epochs):
    print(epoch)
    for batch in train_iterator:
        text = batch.text
        label = batch.label
        print(text, label)
        score = model(text)
        loss = criterion(score, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
