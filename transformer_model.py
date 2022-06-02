import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import spacy

text_eng = torchtext.data.Field(sequential=True, use_vocab=True, init_token='<sos>', eos_token='<eos>',
                                tokenize='spacy', tokenizer_language='en_core_web_sm')

text_ger = torchtext.data.Field(sequential=True, use_vocab=True, init_token='<sos>', eos_token='<eos>',
                                tokenize='spacy', tokenizer_language='de_core_news_sm')

train, val, test = torchtext.datasets.Multi30k.splits(root='.data', exts=('.de', '.en'), fields=(text_ger, text_eng))

batch_size = 128
vocab_size = 10000
learning_rate = 3e-4
num_epochs = 10

text_ger.build_vocab(train, max_size=vocab_size)
text_eng.build_vocab(train, max_size=vocab_size)
train_load = torchtext.data.BucketIterator(train, batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src))

for i in train_load:
    print(i)
    break

sample = "ein pferd geht unter einer brücke neben einem boot."


def translate(model, sentence):
    ger = spacy.load("de_core_news_sm")
    tokens = [token.text for token in ger(sentence)]
    tokens.insert(0, text_ger.init_token)
    tokens.append(text_ger.eos_token)
    # print(tokens)
    txt_to_idx = [text_ger.vocab.stoi[i] for i in tokens]
    idx_to_tensor = torch.LongTensor(txt_to_idx).unsqueeze(1)
    # print(txt_to_idx)
    outputs = [text_eng.vocab.stoi[text_eng.init_token]]

    for _ in range(len(txt_to_idx)):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1)

        with torch.no_grad():
            score = model(idx_to_tensor, trg_tensor)

        pred = score.argmax(2)[-1, :].item()
        outputs.append(pred)

        if pred == [text_eng.vocab.stoi[text_eng.eos_token]]:
            break
    final = [text_eng.vocab.itos[i] for i in outputs]
    return final[1:]


class Network(nn.Module):
    def __init__(self, src_pad_index, src_vocab_size, trg_vocab_size, embed_dim, max_length, num_heads,
                 num_encoder_layers, num_decoder_layers, forward_expansion, dropout):
        super(Network, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.src_pos_embed = nn.Embedding(max_length, embed_dim)
        self.trg_embed = nn.Embedding(trg_vocab_size, embed_dim)
        self.trg_pos_embed = nn.Embedding(max_length, embed_dim)

        self.transformer = nn.Transformer(embed_dim, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout)

        self.fc = nn.Linear(embed_dim, trg_vocab_size)
        self.drops = nn.Dropout(dropout)

        self.src_pad_index = src_pad_index

    def make_mask(self, src):
        src_masked = src.transpose(0, 1) == self.src_pad_index
        return src_masked

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N))
        trg_positions = (torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N))

        src_embed = self.drops(self.src_embed(src) + self.src_pos_embed(src_positions))
        trg_embed = self.drops(self.trg_embed(trg) + self.trg_pos_embed(trg_positions))

        src_masked = self.make_mask(src)
        trg_masked = self.transformer.generate_square_subsequent_mask(trg_seq_len)

        out = self.transformer(src_embed, trg_embed, src_key_padding_mask=src_masked, tgt_mask=trg_masked)
        out = self.fc(out)

        return out


src_vocab_size = len(text_ger.vocab)
trg_vocab_size = len(text_eng.vocab)
n_heads = 6
num_encoder_layers = 3
num_decoder_layers = 3
embed_dim = 216
dropout = 0.1
max_len = 100
forward_expansion = 3
src_pad_index = text_ger.vocab.stoi["<pad>"]

model = Network(src_pad_index, src_vocab_size, trg_vocab_size, embed_dim, max_len, n_heads,
                num_encoder_layers, num_decoder_layers, forward_expansion, dropout)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

idx = text_eng.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=idx)

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    epoch_loss = 0
    model.eval()
    print(translate(model, sample))
    model.train()
    for i, batch in enumerate(train_load):
        if i >= 50 and i % 50 == 0:
            print(f"{i}/{len(train_load)}")
        score = model(batch.src, batch.trg[:-1, :])
        score = score.reshape(-1, score.shape[2])
        batch.trg = batch.trg[1:].reshape(-1)
        print(batch.trg.shape, score.shape)
        loss = criterion(score, batch.trg)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss at epoch {epoch} is: {epoch_loss}")
