"""Small decoder-only character Transformer for learning, not a production LLM."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Config:
    vocab_size:int
    context:int=32
    width:int=32
    heads:int=4
    layers:int=2
    def __post_init__(self):
        if min(self.vocab_size,self.context,self.width,self.heads,self.layers)<1 or self.width%self.heads:
            raise ValueError('Positive dimensions required; width must divide by heads')

class Attention(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.heads=cfg.heads;self.dim=cfg.width//cfg.heads
        self.qkv=nn.Linear(cfg.width,3*cfg.width);self.proj=nn.Linear(cfg.width,cfg.width)
        self.register_buffer('mask',torch.tril(torch.ones(cfg.context,cfg.context,dtype=torch.bool)))
    def forward(self,x):
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q,k,v=[a.view(b,t,self.heads,self.dim).transpose(1,2) for a in (q,k,v)]
        weights=(q@k.transpose(-2,-1))/math.sqrt(self.dim)
        weights=weights.masked_fill(~self.mask[:t,:t],float('-inf')).softmax(dim=-1)
        return self.proj((weights@v).transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.ln1=nn.LayerNorm(cfg.width);self.attn=Attention(cfg)
        self.ln2=nn.LayerNorm(cfg.width)
        self.ff=nn.Sequential(nn.Linear(cfg.width,4*cfg.width),nn.GELU(),nn.Linear(4*cfg.width,cfg.width))
    def forward(self,x):
        x=x+self.attn(self.ln1(x));return x+self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.token=nn.Embedding(cfg.vocab_size,cfg.width);self.position=nn.Embedding(cfg.context,cfg.width)
        self.blocks=nn.Sequential(*[Block(cfg) for _ in range(cfg.layers)])
        self.norm=nn.LayerNorm(cfg.width);self.head=nn.Linear(cfg.width,cfg.vocab_size)
    def forward(self,ids,targets=None):
        if ids.ndim!=2 or not 1<=ids.shape[1]<=self.cfg.context:raise ValueError('Invalid context length')
        x=self.token(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        logits=self.head(self.norm(self.blocks(x)))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,self.cfg.vocab_size),targets.reshape(-1))
        return logits,loss
    @torch.no_grad()
    def generate(self,ids,count=80,temperature=1.0):
        if not 0<=count<=1000 or temperature<=0:raise ValueError('Invalid generation parameters')
        self.eval()
        for _ in range(count):
            logits,_=self(ids[:,-self.cfg.context:]);probs=(logits[:,-1]/temperature).softmax(-1)
            ids=torch.cat([ids,torch.multinomial(probs,1)],dim=1)
        return ids
