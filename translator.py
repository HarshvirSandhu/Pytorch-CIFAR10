import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import spacy
import random


def translate_string(model, english, german, text, max_size=40):
    spacy_ger = spacy.load("de_core_news_sm")
    tokens = [tok.text.lower() for tok in spacy_ger(text)]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    text_to_index = [german.vocab.stoi[t] for t in tokens]

    text_vector = torch.LongTensor(text_to_index).unsqueeze(1)

    with torch.no_grad():
        hidden, cell = model.encoder(text_vector)
    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_size):
        previous_word = torch.LongTensor([outputs[-1]])

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


eng_text = torchtext.data.Field(sequential=True, use_vocab=True, init_token="<sos>", eos_token="<eos>",
                                tokenize="spacy", tokenizer_language='en_core_web_sm', lower=True)

ger_text = torchtext.data.Field(sequential=True, use_vocab=True, init_token="<sos>", eos_token="<eos>",
                                tokenize="spacy", tokenizer_language='de_core_news_sm', lower=True)


train, val, test = torchtext.datasets.Multi30k.splits(exts=(".de", ".en"), fields=(ger_text, eng_text), root=".data")

batch_size = 64
vocab_size = 10000

eng_text.build_vocab(train, max_size=vocab_size)
ger_text.build_vocab(train, max_size=vocab_size)
print(len(eng_text.vocab), len(ger_text.vocab))

train_load = torchtext.data.BucketIterator(dataset=train, batch_size=batch_size,
                                           sort_within_batch=True, sort_key=lambda x: len(x.src))

for i in train_load:
    print(i)
    break

print(len(ger_text.vocab), len(eng_text.vocab))


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_dropout):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(num_dropout)
        self.embed_layer = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, dropout=num_dropout)

    def forward(self, x):
        x = self.drop(self.embed_layer(x))
        out, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, num_dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(num_dropout)
        self.embed_layer = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, dropout=num_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        x = self.drop(self.embed_layer(x))
        out, (hidden, cell) = self.lstm(x)
        output = self.fc(out)
        output = output.squeeze(0)
        # print(output.shape)
        return output, hidden, cell


class Network(nn.Module):
    def __init__(self, encoder, decoder):
        super(Network, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, ratio=0.5):
        target_len = target.shape[0]
        batch_size = source.shape[1]
        target_vocab_size = len(eng_text.vocab)

        hidden, cell = self.encoder(source)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        x = target[0]

        for t in range(1, target_len):
            out, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = out
            pred = out.argmax(1)

            x = target[t] if random.random() < ratio else pred
        # print(outputs.shape)
        return outputs


encoder = Encoder(input_dim=len(ger_text.vocab), embed_dim=512, hidden_dim=1024, num_dropout=0.5)
decoder = Decoder(input_dim=len(eng_text.vocab), embed_dim=512, hidden_dim=1024,
                  num_dropout=0.5, output_dim=len(eng_text.vocab))

model = Network(encoder=encoder, decoder=decoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(20):
    epoch_loss = 0

    model.eval()

    translated_sentence = translate_string(
        model=model, text=sentence, german=ger_text, english=eng_text
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()
    for i, batch in enumerate(train_load):

        if i+1 >= 50 and (i+1) % 50 == 0:
            print(f"{i+1}/{len(train_load)}")

        score = model(batch.src, batch.trg)
        # print("score shape", score.shape)
        score = score[1:].reshape(-1, score.shape[2])
        batch.trg = batch.trg[1:].reshape(-1)
        loss = criterion(score, batch.trg)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss at {epoch+1} is: {epoch_loss}")
    model.eval()

    translated_sentence = translate_string(
        model=model, text=sentence, german=ger_text, english=eng_text
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()


