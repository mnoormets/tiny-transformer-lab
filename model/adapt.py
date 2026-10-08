"""Measured CPU comparison: frozen baseline, LoRA and full fine-tuning."""
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import torch
from .gpt import Config,TinyGPT
from .train import batch,validation
from .lora import attach,save,load,merge
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=150);args=parser.parse_args()
    if not 1<=args.steps<=2000:parser.error('steps must be 1-2000')
    torch.manual_seed(73);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    path=ROOT/'artifacts/checkpoint.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    base=TinyGPT(Config(**checkpoint['config']));base.load_state_dict(checkpoint['state_dict'])
    chars=checkpoint['characters'];vocab={char:index for index,char in enumerate(chars)}
    # Disjoint order IDs; repeated templates are an explicit limitation.
    rows=[f'Kontrolli tellimus {i:03}. Võrdle makset ja laoseisu.\n' for i in range(240)]
    corpus=''.join(rows);unknown=set(corpus)-set(chars)
    if unknown:raise ValueError(f'Unsupported training characters: {unknown}')
    encode=lambda text:torch.tensor([vocab[char] for char in text],dtype=torch.long)
    train=encode(''.join(rows[:180]));held=encode(''.join(rows[180:210]));test=encode(''.join(rows[210:]))
    original=(ROOT/'data/synthetic.txt').read_text(encoding='utf-8');original_held=encode(original[int(len(original)*.8):])
    frozen=validation(base,held);frozen_test=validation(base,test);results={};curves={}
    base_state={key:value.clone() for key,value in base.state_dict().items()}
    for method in ['lora','full']:
        torch.manual_seed(73);model=copy.deepcopy(base)
        if method=='lora':attach(model,rank=4,alpha=8)
        trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=.003);generator=torch.Generator().manual_seed(73)
        started=time.perf_counter();curve=[]
        for step in range(args.steps):
            model.train();x,y=batch(train,model.cfg.context,16,generator);_,loss=model(x,y)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);optimizer.step()
            if step==0 or (step+1)%25==0:curve.append({'step':step+1,'training_loss':loss.item()})
        results[method]={'trainable_parameters':sum(p.numel() for p in trainable),'total_parameters':sum(p.numel() for p in model.parameters()),
            'validation_loss':validation(model,held),'test_loss':validation(model,test),'original_domain_loss':validation(model,original_held),'seconds':time.perf_counter()-started}
        curves[method]=curve
        if method=='lora':
            # All frozen base weights must be exactly unchanged, not just similar.
            for key,value in base_state.items():
                mapped=key.replace('.attn.qkv.','.attn.qkv.base.').replace('.attn.proj.','.attn.proj.base.')
                assert torch.equal(value,model.state_dict()[mapped]),key
            adapter_path=ROOT/'artifacts/adapter.pt';save(model,adapter_path)
            restored=load(base,adapter_path);merged=merge(model);x,y=batch(test,model.cfg.context,4,torch.Generator().manual_seed(9))
            a=model.eval()(x)[0];b=restored.eval()(x)[0];c=merged.eval()(x)[0]
            assert torch.equal(a,b)
            error=(a-c).abs().max().item();assert torch.allclose(a,c,atol=2e-5,rtol=2e-5)
            results[method].update(base_weights_unchanged=True,adapter_roundtrip_exact=True,merge_max_logit_error=error,
                adapter_file_bytes=adapter_path.stat().st_size,rank=4,alpha=8)
    report={'scope':'Real low-rank CPU adaptation of our own tiny synthetic character model; not QLoRA or pretrained production LLM experience.',
        'seed':73,'steps':args.steps,'python':platform.python_version(),'torch':torch.__version__,'device':'cpu',
        'base_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'adaptation_corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest(),
        'split':'order IDs 0-179 train, 180-209 validation, 210-239 final test; shared sentence template',
        'base_validation_loss':frozen,'base_test_loss':frozen_test,'base_original_domain_loss':validation(base,original_held),
        'methods':results,'curves':curves,'limitation':'Tiny model and repeated templates; no general language, instruction following, GPU memory or business impact claim.'}
    (ROOT/'adapter-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='curves'},indent=2))
if __name__=='__main__':main()
