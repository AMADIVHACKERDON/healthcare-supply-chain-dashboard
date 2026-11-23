import os
import urllib.request
import re
import numpy
import matplotlib.pyplot as plt
import torch
#import tiktoken
from torch.utils.data import Dataset, DataLoader
#text = "Hello, World. This, is  a  test"

if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
        file_path = 'the-verdict.txt'
        urllib.request.urlretrieve(url,file_path)

with open('the-verdict.txt', "r", encoding='utf-8') as f:
        raw_text = f.read()
        #print(len(raw_text))

#enc_text = tokenizer.encode(raw_text)

#tokenizer = tiktoken.get_encoding("gpt2")
#tokenizer.encode("Hello World")
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result
#print((preprocessed[:10]))
#preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

#output_dim = 256
#print(vocab_size)
#all_tokens = sorted(list(set(preprocessed)))
#all_tokens.extend(["<|endoftext|>", "<|unk|>"])
#vocab = {token:integer for integer,token in enumerate(all_tokens)}
#print(len(vocab.items())) 

#for i, item in enumerate(list(vocab.items())[-5:]):
 #       print(item)

vocab = {token: integer for integer,token in enumerate(all_words)}
#print(len(vocab))
#int_to_str = {i:s for s,i in vocab.items()}
#print (int_to_str)
#ids = [vocab[s] for s in preprocessed]
#print(ids)


class SimpleTokenizers:
        def __init__(self,vocab):
                self.str_to_int = vocab
                self.int_to_str = {i:s for s,i in vocab.items()}
        
        def encode(self,text, allowed_special=None):
                preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

                preprocessed = [
                        item.strip() for item in preprocessed if item.strip()
                                ]
                preprocessed = [
                        item if item in self.str_to_int
                        else "<|unk|>" 
                        for item in preprocessed
                ]
                ids = [self.str_to_int.get(s, self.str_to_int.get("<|unk|", 0)) for s in preprocessed]
                return ids
        
        def decode(self, ids):
                text = " ".join([self.int_to_str.get(int(i), "<|unk|>") for i in ids])
                text = re.sub(r'\s+([,.?!"()\'])', r'\1',text)
                return text

tokenizer = SimpleTokenizers(vocab)

#text = "Hello, do you like tea. is this-- a test?"
ids = tokenizer.encode(raw_text)
ids_sam = ids[50:]
vocab_size = len(vocab)
#print(len(ids))
class GPTDataset(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
                self.input_ids = []
                self.target_ids = []

                token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

                for i in range(0, len(token_ids) - max_length, stride):
                        input_chunk = token_ids[i:i + max_length]
                        target_chunk = token_ids[i + 1: i + max_length + 1]
                        self.input_ids.append(torch.tensor(input_chunk))
                        self.target_ids.append(torch.tensor(target_chunk))
        
        def __len__(self):
                return len(self.input_ids)
        def __getitem__(self, index):
                return self.input_ids[index],self.target_ids[index]
        

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
        
        tokenizer = SimpleTokenizers(vocab)
       
        dataset = GPTDataset(txt, tokenizer,max_length, stride)

        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers
        )

        return dataloader
max_length = 4
dataloader =create_dataloader_v1(raw_text, batch_size=8, max_length=4,stride=max_length, shuffle=False)

data_iter = iter(dataloader)
batch, targets = next(data_iter)
batch = batch.long()
#print(input[2])
#print("Token IDs:\n", input)
#print("\nInputs Shape:\n", input.shape)
#print(torch.dot(input[2], input[1]))
#second_batch = next(data_iter)
#print(second_batch)

#input_query = input[1]
n = batch.size(0)

#attn_scr_2 = torch.zeros(n, n)
attn_scr_2 = torch.empty(n,n)
for i, x_1 in enumerate(batch):
        for j, x_j in enumerate(batch):
                attn_scr_2[i, j] = torch.dot(x_1, x_j)
#print(attn_scr_2.sum(1))

x_2 = batch[1]

d_in = batch.shape[1]
d_out = 2
#print(input.shape)

W_query = torch.nn.Parameter(torch.rand(d_in,d_out))
W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in,d_out))
#print(d_in)
#print(x_2)
#print(W_query)
#print(W_value.shape)
#print(W_key.shape)
context_length = attn_scr_2.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#print(mask_simple)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scr_2.masked_fill(mask.bool(), -torch.inf)
#print(masked)

layer = torch.nn.Dropout(0.5)
#print(layer(attn_scr_2))
#query_2 = x_2 @ W_query 
#print (query_2)

class SelfAttention(torch.nn.Module):
        def __init__(self, d_in, d_out):
                super().__init__()
                self.W_query = torch.nn.Parameter(torch.rand(d_in,d_out))
                self.W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
                self.W_value = torch.nn.Parameter(torch.rand(d_in,d_out))
        
        def forward(self,x):
                queries = x @ self.W_query
                key = x @ self.W_key
                values = x @ self.W_value
                d_k = key.shape[1]
                attn_scr = queries @ key.transpose(1,2)
                attn_weights = torch.softmax(attn_scr / d_k**0.5, dim=1)
                context_vec = attn_weights @ values

                return context_vec
torch.manual_seed(123)
sa_v1 = SelfAttention(d_in, d_out)
embedding_layer = torch.nn.Embedding(vocab_size, d_in)
emb = embedding_layer(batch)
#print(sa_v1(emb))

class CasualAttention(torch.nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
                super().__init__()
                self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
                self.W_key = torch.nn.Linear(d_in,d_out, bias=qkv_bias)
                self.W_value = torch.nn.Linear(d_in,d_out, bias=qkv_bias)
                self.dropout = torch.nn.Dropout(dropout)
                self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
                x = x.float()
                b, num_tokens, d_in = x.shape
                
                queries = self.W_query(x)
                keys = self.W_key(x)
                values = self.W_value(x)
                
                attn_scr_2 = queries @ keys.transpose(1, 2)
                attn_scr_2.masked_fill(
                        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
                attn_weight = torch.softmax(attn_scr_2/keys.shape[-1]**0.5, dim=1
                )
                attn_weight = self.dropout(attn_weight)
                context_vec = attn_weight @ values
                return context_vec
              

#print(context_vec)
torch.manual_seed(789)

dropout = 0.0
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, dropout)
embedding_layer = torch.nn.Embedding(vocab_size,d_in)
batch = batch.long()
emb = embedding_layer(batch)
#print(ca(emb)) 
#print(context_vec)          
#print(batch)
class MultiHeadAttentionWrapper(torch.nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
                super().__init__()
                assert (d_out % num_heads == 0) 
                self.dropout = torch.nn.Dropout(dropout)
                self.d_out = d_out
                self.num_heads = num_heads
                self.head_dim = d_out // num_heads
                self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
                self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
                self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
                self.out_proj = torch.nn.Linear(d_out, d_out)
                self.register_buffer(
                        "mask", 
                        torch.triu(torch.ones(context_length, context_length), 
                                   diagonal=1)
                )
                 
        def forward(self, x):
                b, num_tokens, d_in = x.shape
                keys = self.W_key(x)
                query = self.W_query(x)
                values = self.W_value(x)
                
                keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
                values = values.view(b, num_tokens, self.num_heads, self.head_dim)
                query = query.view(b, num_tokens, self.num_heads, self.head_dim)

                keys = keys.transpose(1,2)
                query = query.transpose(1,2)
                values = values.transpose(1,2)

                attn_scr_2 = query @ keys.transpose(2,3)

                mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

                attn_scr_2.masked_fill(mask_bool, -torch.inf)
                attn_weight = torch.softmax(attn_scr_2 / keys.shape[-1]**0.5, dim=-1)
                attn_weight = self.dropout(attn_weight)

                context_vec = (attn_weight @ values).transpose(1,2)
                context_vec = context_vec.reshape(b, num_tokens, self.d_out)
                context_vec = self.out_proj(context_vec)
                return context_vec
               
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = batch.shape[-1], 2
embedding_layer = torch.nn.Embedding(vocab_size,d_in)
batch = batch.long()
emb = embedding_layer(batch)
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
#print(mha(emb))
#input * input.T
#attn_scr_2 = torch.empty(input.shape(0))

#for i, x_1 in enumerate(input): 
 #       attn_scr_2[i][i_2] = torch.dot(x_1, input_query)


attn_weight = torch.softmax(attn_scr_2, dim=1)
#all_context = attn_weight * input
#print(attn_weight)
#def softmax_n(x):
 #       return torch.exp(x) / torch.exp(x).sum(dim=0)


#oftmax_n(attn_scr_2)
#print(len(ids))
GPT_CONFIG_124M = {
        "vocab_size": len(ids),
        "context_length": 256,
        "output_dim":  256,
        "n_heads":  16,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
}
class GPTModel(torch.nn.Module):
        def __init__(self,cfg):
                super().__init__()
               
                self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["output_dim"])
                self.pos_emb = torch.nn.Embedding(cfg["context_length"],cfg["output_dim"])
                self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
                self.trf_bocks = torch.nn.Sequential(
                        *[TransfBlck(cfg) for _ in range(cfg["n_layers"])])
                        
                self.final_norm = NormLay(cfg["output_dim"])
                self.out_head = torch.nn.Linear(
                        cfg["output_dim"], cfg["vocab_size"], bias=False
                )
                print("tok_emb dim:", self.tok_emb.embedding_dim)
                print("pos_emb dim:", self.pos_emb.embedding_dim)
                
        def forward(self, in_dx):
               
                
                B, T  = in_dx.shape

                tok_emb = self.tok_emb(in_dx)
                
                pos = torch.arange(T, device=in_dx.device)
                pos_emb = self.pos_emb(pos)
                pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)

                tok_emb = tok_emb[:, :T]

                x = tok_emb + pos_emb
                x = self.drop_emb(x)
                x = self.trf_bocks(x)
                x = self.final_norm(x)
                logits = self.out_head(x)
                return logits

class TransfBlck(torch.nn.Module):
        def __init__(self, cfg):
                super().__init__()
                self.att = MultiHeadAttentionWrapper(
                        d_in=cfg["output_dim"],
                        d_out=cfg["output_dim"],
                        
                       context_length=cfg["context_length"],
                        num_heads=cfg["n_heads"],
                        dropout=cfg["drop_rate"],
                        qkv_bias=cfg["qkv_bias"]
                )                                                                             
                self.ff = FeedFwd(cfg)
                self.norm1 = NormLay(cfg["output_dim"])
                self.norm2 = NormLay(cfg["output_dim"])
                self.drop_shortcut = torch.nn.Dropout(cfg["drop_rate"])
        def forward(self, x):
                shortcut = x
                x = self.norm1(x)
                x = self.att(x)
                x = self.drop_shortcut(x)
                x = x + shortcut

                shortcut = x
                x = self.norm2(x)
                x = self.ff(x)
                x = self.drop_shortcut(x)
                x = x + shortcut

                return x 

class NormLay(torch.nn.Module):
        def __init__(self, output_dim):
                super().__init__()
                self.eps = 1e-5
                self.scale =torch.nn.Parameter(torch.ones(output_dim))
                self.shift = torch.nn.Parameter(torch.zeros(output_dim))

        def forward(self, x):
                mean = x.mean(dim =-1, keepdim =True)
                var = x.var(dim=-1, keepdim =True, unbiased = False)
                norm_x = (x - mean) / torch.sqrt(var + self.eps)
                return self.scale + norm_x + self.shift

class GeLu(torch.nn.Module):
        def __init__(self):
                super().__init__()
        def forward(self, x):
                return 0.5 * x * (1 + torch.tanh(
                        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                        (x + 0.044715 * torch.pow(x, 3))
                ))

gelu, relu = GeLu(), torch.nn.ReLU()

#x = torch.linspace(-3, 3, 100)
#y_gelu, y_relu = gelu(x), relu(x)
#for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
 #       plt.subplot(1, 2, 1)
 #       plt.plot(x,y)
 #       plt.title("label activation function")
  #      plt.xlabel("x")
   #     plt.ylabel(f"(label)(x)")
    #    plt.grid(True)

#plt.tight_layout()
#plt.show()
class FeedFwd(torch.nn.Module):
        def __init__(self, cfg):
                super().__init__()
                self.layers = torch.nn.Sequential(
                        torch.nn.Linear(cfg["output_dim"], 4 * cfg["output_dim"]),
                        GeLu(),
                        torch.nn.Linear(4 * cfg["output_dim"], cfg["output_dim"])
                )
        def forward(self,x):
                return self.layers(x)

ffn = FeedFwd(GPT_CONFIG_124M)
x = torch.rand(2,4, 256)
block = TransfBlck(GPT_CONFIG_124M)

output = block(x)
#print(output.shape)
#print(ffn.layers[0].weight)
class DNN(torch.nn.Module):
        def __init__(self, layer_sizes, use_shortcut):
                super().__init__()
                self.use_shortcut = use_shortcut
                self.layers = torch.nn.ModuleList(
                        [
                                torch.nn.Sequential(torch.nn.Linear(layer_sizes[0], layer_sizes[1]), GeLu()),
                                torch.nn.Sequential(torch.nn.Linear(layer_sizes[1], layer_sizes[2]), GeLu()),
                                torch.nn.Sequential(torch.nn.Linear(layer_sizes[2], layer_sizes[3]), GeLu()),
                                torch.nn.Sequential(torch.nn.Linear(layer_sizes[3], layer_sizes[4]), GeLu()),
                                torch.nn.Sequential(torch.nn.Linear(layer_sizes[4], layer_sizes[5]), GeLu())
                        ]
                )
        def forward(self, x):
                for layer in self.layers:
                        layer_output = layer(x)
                        if self.use_shortcut and x.shape == layer_output.shape:
                                x = x + layer_output
                        else:
                                x = layer_output
                return x
def print_grad(model,x):
        x = x.float()
        output = model(x)
        target = torch.zeros(output.shape).float()

        loss = torch.nn.MSELoss()
        loss = loss(output, target)

        loss.backward()
        for name, param in model.named_parameters():
                if 'weight' in name:
                        print(f"name has gradient mean of{param.grad.abs().mean().item()}")
                

torch.manual_seed(123)
layer_sizes = [3, 3, 3, 3, 3, 1]
model_without_shortcut = DNN(layer_sizes, use_shortcut=False)
batch = torch.rand(8,3).float()

#print_grad(model_without_shortcut, batch)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
#print(batch.shape)
batch = batch.long()
logits = model(batch)
#print("Output shape:", logits.shape)
#print(logits.shape)
total_params = sum(p.numel() for p in model.parameters())
#print(total_params)
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
                idx_cond = idx[:, -context_size:]
                with torch.no_grad():
                        logits = model(idx_cond)
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
       
def text_to_token_ids(text, tokenizer):
        encoded= tokenizer.encode(text, allowed_special = ("<|endoftext|>"))
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())

token_ids = generate(
        model= model,
        idx= text_to_token_ids(raw_text, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4

)
#print(token_ids.squeeze(0).shape)
print(token_ids)
#print(token_ids_to_text(token_ids, tokenizer))
probas = torch.softmax(logits, dim=1)
token_ids = torch.argmax(probas, dim=1, keepdim=True)
print("Token IDs:\n", token_ids)
print(f"Target batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
print("target_shape:",  targets.shape)
print("probas_shape:",  probas.shape)
total_text = len(raw_text)
total_token = tokenizer.encode(raw_text)
print(total_text)
print(total_token)
text_idx = 0
text_idx_2 = 1
seq_len = probas.shape[1]
target_probas = probas[text_idx, torch.arange(seq_len), targets[text_idx, :seq_len]]
target_probas_2 = probas[text_idx_2, torch.arange(seq_len), targets[text_idx, :seq_len]]
print("targets[text_idx]:", targets[text_idx])
print(target_probas)

log_probas = torch.log(torch.cat((target_probas,target_probas_2)))
print(log_probas)

print(torch.mean(log_probas))

train_ratio = 0.90
split_idx = int(train_ratio * total_text)
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]
train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
)

val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
)
train_tokens = 0
for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()
val_tokens = 0
for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()
        


print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", val_tokens + train_tokens)

def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
                return float("nan")
        elif num_batches is None:
                num_batches = len(data_loader)
        else:
                num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
                if i < num_batches:
                        loss = calc_loss_batch(input_batch, target_batch, model, device)
                        total_loss += loss.item()
                else:
                        break
        return total_loss/ num_batches
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
        train_losses, val_losses, track_tokn_sn = [], [], []
        token_sn, glo_step = 0, -1

        for epoch in range(num_epochs):
                model.train()

                for input_batch, target_batch in train_loader:
                        optimizer.zero_grad()
                        loss = calc_loss_batch(input_batch, target_batch, model, device)
                        loss.backward()
                        optimizer.step()
                        token_sn += input_batch.numel()
                        glo_step += 1

                        if glo_step % eval_freq == 0:
                                train_loss, val_loss = evaluate_model(
                                        model, train_loader,val_loader, device, eval_iter
                                )
                                train_losses.append(train_loss)
                                val_losses.append(val_loss)
                                track_tokn_sn.append(token_sn)
                                print(f"Ep {epoch+1} (Step {glo_step:06d}): "
                                      f"Train loss {train_loss:.3f}, val loss {val_loss:.3f}")
                
                generate_and_print_sample(
                        model, tokenizer, device, start_context
                )
        
        return train_losses, val_losses, track_tokn_sn

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
                token_ids = generate(
                        model=model, idx=encoded,
                        max_new_tokens=50, context_size= context_size,
                        top_k=25,
                        temperature=1.4
                )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
        model.train()
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
#train_losses, val_losses, token_sn = train_model_simple(
   #     model, train_loader, val_loader, optimizer, device,
  #      num_epochs=num_epochs, eval_freq=5, eval_iter=5,
 #       start_context= raw_text, tokenizer=tokenizer
#)
#from matplotlib.ticker import MaxNLocator
#def plot_losses(epochs_seen, token_sn, train_losses, val_losses):
       # fig, ax1 = plt.subplots(figsize=(5,3))

       # ax1.plot(epochs_seen,train_losses, label = "Training loss")
       # ax1.plot(epochs_seen, val_losses, linestyle="-", label="Validation loss")
       # ax1.set_xlabel("Epochs")
       # ax1.set_ylabel("Loss")
       # ax1.legend(loc="upper right")
       # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        #ax2 = ax1.twiny()
        #ax2.plot(token_sn, train_losses, alpha=0)
       # ax2.set_xlabel("Token seen")

        #fig.tight_layout()
        #plt.savefig("loss-plot.pdf")
        #plt.show()

#epochs_tensor = torch.linspace(0, num_epochs , len(train_losses))
#plot_losses(epochs_tensor, token_sn, train_losses, val_losses)
print(probas.shape)
def print_sampled_tokens(probas):
        torch.manual_seed(123)
        probas = probas[0, -1, :] 
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
        sample_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sample_ids):
                print(f"{freq} x {tokenizer.decode([i])}")
print_sampled_tokens(probas)
def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=-1)

temperature = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(logits, T) for T in temperature]

top_k = 3
logits = logits[0,-1, :]
top_logits = torch.topk(logits, top_k)
new_logits = torch.where(
        condition=logits < top_logits.values[-1],
        input= torch.tensor(float("-inf")),
        other= logits
)


print(new_logits)
#encoded = tokenizer.encode(start_context)

#encoded_tensor = torch.tensor(encoded).unsqueeze(0)
#output_ids = encoded + start_context.tolist()
#output = generate_text(
 #       model= model,
  #      idx= encoded_tensor,
   #     max_new_tokens= 4,
    #    context_size= GPT_CONFIG_124M["context_length"]
#)
#print(output)
#output_id = output.squeeze(0).tolist()
#print(tokenizer.decode(output_id))
#print(tokenizer.decode(output_ids))
#query = batch[1]
#context_vec = torch.zeros(query.shape)
#for i,x_1 in enumerate (input):
     #   context_vec += attn_scr_2[i]*input[i]
#print(context_vec) 
#vocab_size = len(ids)
#output_dim = 256
#embedding_layer =  torch.nn.Embedding(vocab_size, output_dim)
#token_embedding = embedding_layer(input)

#context_length = max_length
#pos_embedd = torch.nn.Embedding(context_length,output_dim)



#print(pos_embedd.weight)
#pos_embeddlay = pos_embedd(torch.arange(max_length))
#print(pos_embeddlay)
#print(pos_embeddlay.shape)
#print(token_embedding.shape)

#input_embeddings = token_embedding + pos_embeddlay
#print(input_embeddings)
#print(embedding_layer.weight)
#context_size = 4

#x = ids_sam[:context_size]
#y = ids_sam[1:context_size+1]

#print(f"x: {x}")
#print(f"y:     {y}")

#for i in range(1, context_size+1):
 #       context = ids_sam[:i]
  #      desired = ids_sam[i]
   #     print(tokenizer.decode(context), "------>", tokenizer.decode([desired]))
#print(len(ids))
#all_tokens = sorted(list(set(preprocessed)))
#all_tokens.extend(["<[endoftext]>", "[unk]>"])

#vocab = (token:integer for integer,token in enumerate[all_tokens])      

#print(ids)
#print(tokenizer.decode(ids))
#print(len(vocab.items()))       
#for i, item in enumerate(list(vocab.items())[:5]):
 #       print(item)