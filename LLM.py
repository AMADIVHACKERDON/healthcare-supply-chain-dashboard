
import random
import torch

import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('Hello Data Buddies blushgopher.txt', 'r') as f:
    text = f.read().lower()[:50000]

chars = sorted(set(text))
vocab_size = len(chars)

char2idx = {c:i for i, c in enumerate(chars)}
idx2char = {i:c for c, i in char2idx.items()}

seq_length = 100
step_size = 1
data = [(text[i:i+seq_length], text[i+seq_length]) for i in range(0, len(text)-seq_length, step_size)]

X = torch.tensor([[char2idx[c] for  c in seq] for seq, _ in data]).to(device)
Y = torch.tensor([char2idx[c] for _, c in data]).to(device)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x,hidden)
        out =self.fc(out[:, -1, :])

        return out, hidden
model = CharLSTM(vocab_size, hidden_size=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    running_loss = 0.0

    for i in range(0,len(X), 64):
        x_batch = X[i:i+64].to(device)
        y_batch = Y[i:i+64].to(device)

        if len(x_batch) == 0:
            continue

        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/20, Loss: {running_loss:4f}')
print(output)

def generate(model, start_seq, length=200, temperature=0.0, top_k=None, eos_id=None):
        model.eval()
        input_seq = torch.tensor([[char2idx[c] for c in start_seq]]).to(device)
        for _ in range(length):
                logits = model(input_seq)
                logits = logits[:, -1, :]
                probas = torch.softmax(logits, dim=-1)
                if top_k is not None:
                        top_logits, top_po = torch.topk(logits, top_k)
                        new_logits = torch.where(
                                condition=logits < top_logits[:, -1],
                                input=torch.tensor(float("-inf")),
                                other=logits
                                )
                
                if temperature > 0.0:
                        logits = logits / temperature
                        probas = torch.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probas, num_samples=1)
                else:
                        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
                
                if idx_next == eos_id:
                        break
                idx = torch.cat((idx, idx_next), dim=1)
                #return probas
                
        return idx
       
print(generate(model, 'raw marketing data for analysis by addressing data quality issues and creating a new feature'))
print(generate(model, 'for further marketing analysis while introducing a new short_title feature for concise product titles'))
print(generate(model, 'significant variables for cleaning and for creating the short_title feature'))

torch.save(model.state_dict(), 'lstm_model.pth')