import transformers
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

model_name = "bert-base-uncased"
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
# Freezing all weights except last layer
for step, p in enumerate(model.parameters()):
    if 200-step>0:
        p.requires_grad=False

tokeniser = transformers.AutoTokenizer.from_pretrained(model_name)
text = 'A connection to the notebook server could not be established. The notebook will continue trying to reconnect.' \
         'Check your network connection or notebook server configuration.'

print(tokeniser.tokenize(text))
tokens = tokeniser.tokenize(text)
token_ids = tokeniser.convert_tokens_to_ids(tokens)
print(token_ids)

batch = tokeniser(text, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    out = model(**batch)
    print(F.softmax(out.logits))


df = pd.read_csv('C:/Users/harsh/Downloads/SpamClassifier-master/SpamClassifier-master/smsspamcollection/SMSSpamCollection', sep='\t')
labels = le.fit_transform(df['label'])
df['transformed_labels'] = labels
train_text, test_text, train_label, text_label = train_test_split(df['text'], df['transformed_labels'], test_size=0.2)

train_tokens = tokeniser.batch_encode_plus(train_text.tolist(), max_length=300, padding=True, truncation=True)
train_seq = torch.tensor(train_tokens['input_ids'])
train_mask = torch.tensor(train_tokens['attention_mask'])
train_label = torch.tensor(train_label.tolist())

Data = TensorDataset(train_seq, train_mask, train_label)

data_load = DataLoader(Data, 32)

for i in data_load:
    print(i)
    break


class BERT_Arch(torch.nn.Module):
    def __init__(self, bert, num_classes):
        super(BERT_Arch, self).__init__()

        self.bert = bert
        self.fc1 = torch.nn.Linear(768, num_classes)
        self.bert.classifier = self.fc1
        self.softmax = torch.nn.Softmax()

    def forward(self, sent_id, mask):
        x = self.bert(sent_id, attention_mask=mask)
        # print(x)
        x = self.softmax(x.logits)
        return x


model = BERT_Arch(model, num_classes=2)
optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
for seq, mask, label in data_load:
    score = model(seq, mask)
    print(score, label)
    loss = criterion(score, label)
    print(loss)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    break