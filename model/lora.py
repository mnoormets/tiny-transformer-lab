"""Low-rank adaptation with frozen base weights and verifiable adapter provenance."""
import copy,hashlib,json,math
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F

class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__()
        if not isinstance(base,nn.Linear) or type(rank) is not int or not 1<=rank<=min(base.in_features,base.out_features):raise ValueError('Invalid rank or base layer')
        if not math.isfinite(alpha) or alpha<=0:raise ValueError('Invalid alpha')
        self.base=base;self.rank=rank;self.alpha=float(alpha)
        for parameter in base.parameters():parameter.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b)*(self.alpha/self.rank)
    def merged(self):
        result=copy.deepcopy(self.base)
        with torch.no_grad():result.weight.add_((self.b@self.a)*(self.alpha/self.rank))
        return result

def base_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tensor.shape).encode());digest.update(str(tensor.dtype).encode())
        digest.update(bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()

def attach(model,rank=4,alpha=8):
    if any(isinstance(layer,LoRALinear) for layer in model.modules()):raise ValueError('Adapters already attached')
    if type(rank) is not int or not 1<=rank<=model.cfg.width or not math.isfinite(alpha) or alpha<=0:raise ValueError('Invalid adapter configuration')
    identity=base_hash(model)
    for parameter in model.parameters():parameter.requires_grad_(False)
    targets=[]
    for index,block in enumerate(model.blocks):
        for attribute in ['qkv','proj']:
            name=f'blocks.{index}.attn.{attribute}'
            setattr(block.attn,attribute,LoRALinear(getattr(block.attn,attribute),rank,alpha));targets.append(name)
    model.adapter_metadata={'format':1,'base_sha256':identity,'config':asdict(model.cfg),'rank':rank,'alpha':float(alpha),'targets':targets}
    return model

def adapter_state(model):
    return {key:tensor.detach().cpu().clone() for key,tensor in model.state_dict().items() if key.endswith(('.a','.b'))}

def save(model,path):
    torch.save({'metadata':model.adapter_metadata,'state':adapter_state(model)},path)

def load(model,path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    metadata=payload['metadata']
    if metadata['format']!=1 or metadata['base_sha256']!=base_hash(model) or metadata['config']!=asdict(model.cfg):raise ValueError('Adapter/base mismatch')
    # Validate on a copy: a malformed checkpoint cannot partially mutate the caller.
    candidate=attach(copy.deepcopy(model),metadata['rank'],metadata['alpha'])
    if candidate.adapter_metadata!=metadata:raise ValueError('Adapter target mismatch')
    expected=adapter_state(candidate);state=payload['state']
    if set(state)!=set(expected):raise ValueError('Invalid adapter keys')
    for key,value in state.items():
        if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all():raise ValueError('Invalid adapter tensor')
    candidate.load_state_dict(state,strict=False)
    return candidate

def merge(model):
    result=copy.deepcopy(model)
    for block in result.blocks:
        for attribute in ['qkv','proj']:
            layer=getattr(block.attn,attribute)
            if not isinstance(layer,LoRALinear):raise ValueError('Adapter missing')
            setattr(block.attn,attribute,layer.merged())
    if hasattr(result,'adapter_metadata'):del result.adapter_metadata
    return result
