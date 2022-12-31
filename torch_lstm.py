import torch
import torchtext

text = torchtext.data.Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='en_core_web_sm')
label = torchtext.data.LabelField(dtype=torch.long)
fields = [('Label', label), ('Text', text)]
path_data = "C:/Users/harsh/Downloads/SpamClassifier-master/SpamClassifier-master/smsspamcollection/" \
            "SMSSpamCollection"
data_set = torchtext.data.TabularDataset(path=path_data, format='tsv', fields=fields, skip_header=True)
text.build_vocab(data_set, max_size=5000)
label.build_vocab(data_set)

data_load = torchtext.data.BucketIterator(data_set, batch_size=32)


class Model(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.embed = torch.nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim)       # returns hidden_state, cell_state
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=0)
        x = self.linear(x)
        return torch.nn.functional.sigmoid(x)


print(len(text.vocab))
model = Model(input_dim=len(text.vocab), embed_dim=128, hidden_dim=100, output_dim=2)
for data in data_load:
    # data.Text.shape: [len, batch_size], data.Text[i] is not the ith sentence of the batch. BUT data.Text.T[i] is the ith sentence of the batch 
    print(data.Text.shape)
    print(model(data.Text.T).shape)
    print(model(data.Text.T))
    break
