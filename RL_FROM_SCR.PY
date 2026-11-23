import os
import urllib.request
import re

import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from gating import TopKGate
from expert import ExpertMLP


with open('Hello Data Buddies blushgopher.txt', "r", encoding='utf-8') as f:
        raw_text = f.read()

result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token: integer for integer,token in enumerate(all_words)}

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
ids = tokenizer.encode(raw_text)
ids_sam = ids[50:]
vocab_size = len(vocab)


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
        
class BPETokenizer:
        def __init__(self, text: str, block_size: int = 256, split: float =0.9):
                data = self.encode(text)
                data = torch.tensor(list(data), dtype=torch.long)
                n = int(len(data) * split)
                self.train = data[:n]
                self.val = data[n:]
                self.block_size = block_size

        def get_batch(self, which, batch_size: int, device: torch.device):
                buf = self.train if which == "train" else self.val
                assert len(buf) - self.block_size - 1, "file too small for given block size"
                ix = torch.randint(len(buf) - self.block_size, (batch_size,))
                x = torch.stack([buf[i:i+self.block_size] for i in ix])
                y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
                return x.to(device), y.to(device)
        def encode(self,s: str,  allowed_special=None) -> torch.Tensor:
                return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)

        def decode(self, ids) -> str:
                if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                return bytes(ids).decode("utf-8", errors="ignore")
        @property
        def vocab_size(self) -> int:
                return 256
        
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
        
        tokenizer = BPETokenizer(raw_text, block_size=max_length)
       
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
n = batch.size(0)
attn_scr_2 = torch.empty(n,n)
for i, x_1 in enumerate(batch):
        for j, x_j in enumerate(batch):
                attn_scr_2[i, j] = torch.dot(x_1, x_j)

x_2 = batch[1]

d_in = batch.shape[1]
d_out = 2
W_query = torch.nn.Parameter(torch.rand(d_in,d_out))
W_key = torch.nn.Parameter(torch.rand(d_in,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in,d_out))

context_length = attn_scr_2.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scr_2.masked_fill(mask.bool(), -torch.inf)

layer = torch.nn.Dropout(0.5)
embedding_layer = torch.nn.Embedding(vocab_size, d_in)
emb = embedding_layer(batch)

class CasualAttention(torch.nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout,  num_heads, qkv_bias=False):
                super().__init__()
                self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
                self.W_key = torch.nn.Linear(d_in,d_out, bias=qkv_bias)
                self.W_value = torch.nn.Linear(d_in,d_out, bias=qkv_bias)
                self.dropout = torch.nn.Dropout(dropout)
                self.num_heads = num_heads
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

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPT_CONFIG_124M = {
        "vocab_size": len(ids),
        "batch_size": 2,
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
                self.att = CasualAttention(
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
model = GPTModel(GPT_CONFIG_124M)
model = torch.compile(model)
model.eval()
dx = BPETokenizer(raw_text, block_size=GPT_CONFIG_124M["context_length"])
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
def estimate_loss(model: GPTDataset, dx: BPETokenizer, GPT_CONFIG_124M) -> dict:
        model.eval()
        out = {}
        with torch.no_grad():
                for split in ['train', 'val']:
                        losses = []
                        for _ in range(GPT_CONFIG_124M["n_layers"]):
                                xb, yb = dx.get_batch(split, GPT_CONFIG_124M["batch_size"], device)
                                logits = model(xb)

                                loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                yb.view(-1)
                                )
                                

                                losses.append(loss.item())
                        out[split] = sum(losses) / len(losses)
                        print(out)


        model.train()
        
        return out
print(token_ids)
losses = estimate_loss(model, dx, GPT_CONFIG_124M)
print(losses)

class TopKGate(torch.nn.Module):
        def __init__(self, dim, n_expert, k):
                super().__init__()
                assert k = 1 and k = n_expert
                self.n_expert = n_expert
                self.k = k
                self.w_g = torch.nn.Linear(dim, n_expert, bias=True)


        def forward(self, x:torch.Tensor):
                logits = self.w_g(x)
                probas = torch.softmax(logits, dim=-1)
                topk_vals, topk_idx = torch.topk(probas, k=self.k, dim=-1)

                s, e = probas.size(0), probas.size(1)
                importance = probas.mean(dim=0)
                hard1 = topk_idx(1,0)
                load = torch
                load.scatter_add
                load = load/max (5,1)
                max_loss = 

                return topk_idx, topk_vals, max_loss

class MoE(torch.nn.Module):
        def __int__(self, dim, int, n_experts: int, k: int =1, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
                super().__init__()
                self.dim = dim
                self.n_experts = n_experts
                self.k = k
                self.gate = TopKGate(dim, n_experts, k=k)
                self.experts = torch.nn.ModuleList(ExpertMLP:dim, mult=mult, swiglu=swiglu, dropout=dropout, for _ in range(n_experts))

        def forward(self, torch.Tensor):

if __name__ == "__main__":
        p = argparse.ArgumentParse()
        p.add_argument("--top_k", type=int, default=1)
        p.add_argument("--expert", type=int, default=1)
        p.add_argument("--tokens", type=int, default=64)
        p.add_argument("vocab_size", type=int, default=len(ids))
        p.add_argument("batch_size", type=int, default=2)
        p.add_argument("context_length", type=int, default=256)
        p.add_argument("output_dim", type=int, default=256)
        p.add_argument("drop_rate", type=int, default=0.1)
        p.add_argument("qkv_bias", type=int, default=False)
                
        GPT_CONFIG_124M = {
        "vocab_size": len(ids),
        "batch_size": 2,
        "context_length": 256,
        "output_dim":  256,
        "n_heads":  16,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
}

