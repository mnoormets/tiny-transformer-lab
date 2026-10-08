import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import torch
from .gpt import Config,TinyGPT
ROOT=Path(__file__).resolve().parents[1]

def batch(data,context,size,generator):
    if len(data)<=context:raise ValueError('Text too short for a shifted context batch')
    starts=torch.randint(len(data)-context,(size,),generator=generator)
    return torch.stack([data[i:i+context] for i in starts]),torch.stack([data[i+1:i+context+1] for i in starts])

@torch.no_grad()
def validation(model,data):
    model.eval();total=0;count=0
    # Fixed non-overlapping target windows; the held-out suffix is never trained on.
    for start in range(0,len(data)-1,model.cfg.context):
        end=min(start+model.cfg.context,len(data)-1)
        x=data[start:end][None,:];y=data[start+1:end+1][None,:]
        _,loss=model(x,y);total+=loss.item()*(end-start);count+=end-start
    return total/count

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=150);args=parser.parse_args()
    if not 1<=args.steps<=10000:parser.error('steps must be 1-10000')
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    text=(ROOT/'data/synthetic.txt').read_text(encoding='utf-8')
    chars=sorted(set(text));stoi={c:i for i,c in enumerate(chars)}
    split=int(len(text)*0.8);train=torch.tensor([stoi[c] for c in text[:split]],dtype=torch.long)
    held=torch.tensor([stoi[c] for c in text[split:]],dtype=torch.long)
    cfg=Config(vocab_size=len(chars));model=TinyGPT(cfg);optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
    generator=torch.Generator().manual_seed(42)
    counts=torch.bincount(train,minlength=len(chars)).float()+1
    unigram=(counts/counts.sum());baseline=-torch.log(unigram[held[1:]]).mean().item()
    before=validation(model,held);log=[];start=time.perf_counter()
    for step in range(args.steps):
        model.train();x,y=batch(train,cfg.context,16,generator);_,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step==0 or (step+1)%25==0 or step+1==args.steps:log.append({'step':step+1,'training_loss':loss.item()})
    after=validation(model,held)
    output=ROOT/'artifacts';output.mkdir(exist_ok=True)
    torch.save({'config':asdict(cfg),'state_dict':model.state_dict(),'characters':chars},output/'checkpoint.pt')
    sample_ids=model.generate(torch.tensor([[stoi['T']]]),count=120)
    report={'scope':'Tiny CPU learning experiment on authored synthetic text, not model fine-tuning or real-language quality.',
        'seed':42,'steps':args.steps,'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':2,
        'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters()),'corpus_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'split_character':split,'train_characters':len(train),'validation_characters':len(held),
        'unigram_validation_loss':baseline,'initial_validation_loss':before,'final_validation_loss':after,
        'final_validation_perplexity':math.exp(after),'training_seconds':time.perf_counter()-start,'training_curve':log,
        'sample':''.join(chars[i] for i in sample_ids[0].tolist()),
        'limitation':'Train and validation share synthetic templates; this does not demonstrate general language competence.'}
    (ROOT/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='training_curve'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
