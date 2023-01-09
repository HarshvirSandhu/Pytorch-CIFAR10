import torch
import torch.nn as nn
import torchtext
import math
import spacy

data_eng = torchtext.data.Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='en_core_web_sm',
                                init_token='<sos>', eos_token='<eos>', lower=True)
data_ger = torchtext.data.Field(sequential=True, use_vocab=True, tokenize='spacy', tokenizer_language='de_core_news_sm',
                                init_token='<sos>', eos_token='<eos>', lower=True)


train_data, val_data, test_data = torchtext.datasets.Multi30k.splits(exts=('.de', '.en'), fields=(data_ger, data_eng))

data_eng.build_vocab(train_data)
data_ger.build_vocab(train_data)

data_load = torchtext.data.BucketIterator(dataset=train_data, batch_size=64, shuffle=False)
test_load = torchtext.data.BucketIterator(dataset=test_data, batch_size=64, shuffle=False)

# data_load
# [[word1_sentence1, word2_sentence1, ... , word_n_sentence1],
#  [word1_sentence2, word2_sentence2, ... , word_n_sentence2],
#   :
#   :
#   :
#  [word1_sentence_n, word2_sentence_n, ... , word_n_sentence_n]]
# Reason: So that a batch of words at 'n'th position can be used as input to perform operation on words at 'n+1'th position
# Example: predict[batch of words at 'n+1'th position] = model(batch of words at 'n'th position)


for i in data_load:
    print(i.src.T.shape, i.trg.T.shape)
    for word in i.trg.T[-3]:
        print(data_eng.vocab.itos[word], end=' ')
    for word in i.src.T[-3]:
        print(data_ger.vocab.itos[word], end=' ')
    break
print('\n')



class Model(torch.nn.Module):
    def __init__(self, embed_dim_src, embed_dim_trg, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, input_vocab_size, outut_vocab_size, src_pad_idx, max_length, device):
        super(Model, self).__init__()

        self.embed_layer_src = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=embed_dim_src)
        self.embed_layer_trg = torch.nn.Embedding(num_embeddings=outut_vocab_size, embedding_dim=embed_dim_trg)
        self.transformer = torch.nn.Transformer(d_model=embed_dim_src, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                dropout=dropout, device=device, batch_first=False)
        self.linear = torch.nn.Linear(embed_dim_src, outut_vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.embed_dim_src = embed_dim_src
        self.src_pad_idx = src_pad_idx
        self.max_len = max_length

    def positional_encoding(self, x):
        seq_len = x.shape[1]
        # print(seq_len, '----')
        indices = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim_src, 2) * (-math.log(10000.0) / self.embed_dim_src))
        pe = torch.zeros(x.shape[1], 1, self.embed_dim_src)
        pe[:, 0, 0::2] = torch.sin(indices * div_term)
        pe[:, 0, 1::2] = torch.cos(indices * div_term)
        # print(pe.shape, '------------')
        return pe.transpose(0, 1).to(device)

    def make_src_mask(self, src):
        src_mask = src == self.src_pad_idx
        return src_mask

    def forward(self, src, trg):
        # print(src.dtype, src.shape)
        embed_src = self.embed_layer_src(src)
        # print(embed_src.shape, '================',trg.shape)
        embed_src += self.positional_encoding(src)
        embed_src = self.dropout(embed_src)

        embed_trg = self.dropout(self.embed_layer_trg(trg))

        src_mask = self.make_src_mask(src).to(device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[1]).to(device)
        # print(embed_src.transpose(0, 1).shape, embed_trg.transpose(0, 1).shape, src_mask.shape, trg_mask.shape)

        out = self.transformer(embed_src.transpose(0, 1), embed_trg.transpose(0, 1), src_key_padding_mask=src_mask, tgt_mask=trg_mask.T)
        out = self.linear(out)
        return out


loss_list = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_src_dim = 288
embed_trg_dim = 288
nhead = 12
num_encoder_layers = 6
num_decoder_layers = 6
dim_feed_forward = 4
dropout = 0.1
src_vocab_size = len(data_ger.vocab)
trg_vocab_size = len(data_eng.vocab)
src_pad_idx = data_eng.vocab.stoi['<pad>']

print(len(data_load), device)


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]



class TransformerModel(torch.nn.Module):
    def __init__(self, inp_vocab_size, out_vocab_size, embed_dim, dim_feedforward, src_pad_idx,
                 dropout, nhead, num_encoder_layers, num_decoder_layers, device):
        super(TransformerModel, self).__init__()
        self.src_embed = torch.nn.Embedding(inp_vocab_size, embed_dim)
        self.trg_embed = torch.nn.Embedding(out_vocab_size, embed_dim)
        self.transformer = torch.nn.Transformer(d_model=embed_dim, nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.src_pad_idx = src_pad_idx
        self.src_embed_dim = embed_dim
        self.drop = torch.nn.Dropout(dropout)
        self.last = torch.nn.Linear(embed_dim, out_vocab_size)

    def make_src_mask(self, x):
        mask = x.transpose(0, 1) == self.src_pad_idx
        return mask

    def positional_encoder(self, x):
        seq_len = x.shape[0]
        indices = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.src_embed_dim, 2) * (-math.log(10000.0)/self.src_embed_dim))
        pe = torch.zeros_like(x)
        pe[:, 0, 0::2] = torch.sin(indices * div_term)
        pe[:, 0, 1::2] = torch.cos(indices * div_term)
        return pe.to(device)

    def forward(self, src, trg):
        embed_src = self.src_embed(src)
        embed_src +=self.positional_encoder(embed_src)
        embed_src = self.drop(embed_src)

        embed_trg = self.drop(self.trg_embed(trg))

        mask_src = self.make_src_mask(src).to(device)
        mask_trg = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=mask_src, tgt_mask=mask_trg)
        out = self.last(out)
        return out


model = TransformerModel(embed_dim=embed_src_dim, num_encoder_layers=num_encoder_layers,
              num_decoder_layers=num_decoder_layers, nhead=nhead, dim_feedforward=dim_feed_forward, dropout=dropout,
              inp_vocab_size=src_vocab_size, out_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx,
              device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=data_eng.vocab.stoi['<pad>'])
sentence = 'ein Mann, der auf einem Turm neben einem Vogel steht.'
num_epochs = 20
for epoch in range(1, num_epochs):
    epoch_loss = 0
    model.train()
    for num, i in enumerate(data_load):
        i.src = i.src.to(device)
        i.trg = i.trg.to(device)
        output = model(i.src, i.trg[:-1])
        if num >= 200 and num % 200 == 0:
            print(epoch, '---', num)

        output = output.reshape(-1, output.shape[2])
        tgt = i.trg[1:].reshape(-1)
        # print(trg.shape, output.shape)
        loss = criterion(output, tgt)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    loss_list.append(epoch_loss)
    print(epoch, epoch_loss, min(loss_list))
    model.eval()
    print(translate_sentence(model, sentence, data_ger, data_eng, device))