import torch,pytest
from model.gpt import Config,TinyGPT
from model.lora import LoRALinear,attach,adapter_state,save,load,merge

def model():
    torch.manual_seed(4)
    return TinyGPT(Config(vocab_size=12,context=8,width=16,heads=2,layers=1))

def test_initial_adapter_preserves_base_outputs_and_only_adapter_is_trainable():
    base=model();x=torch.tensor([[1,2,3]])
    before=base(x)[0].detach();attach(base)
    assert torch.equal(before,base(x)[0])
    assert all(name.endswith(('.a','.b')) for name,p in base.named_parameters() if p.requires_grad)

def test_merge_and_saved_adapter_reproduce_logits(tmp_path):
    base=model();adapted=attach(model());x=torch.tensor([[1,2,3]])
    with torch.no_grad():
        for module in adapted.modules():
            if isinstance(module,LoRALinear):module.b.normal_(std=.03)
    path=tmp_path/'adapter.pt';save(adapted,path);restored=load(base,path)
    expected=adapted(x)[0]
    assert torch.equal(expected,restored(x)[0])
    assert torch.allclose(expected,merge(adapted)(x)[0],atol=2e-5,rtol=2e-5)

def test_wrong_base_rejected_without_mutation(tmp_path):
    adapted=attach(model());path=tmp_path/'adapter.pt';save(adapted,path)
    wrong=model()
    with torch.no_grad():wrong.head.weight.add_(1)
    with pytest.raises(ValueError,match='mismatch'):load(wrong,path)
    assert not any(isinstance(layer,LoRALinear) for layer in wrong.modules())

def test_malformed_adapter_rejected_without_mutation(tmp_path):
    adapted=attach(model());path=tmp_path/'adapter.pt';save(adapted,path)
    payload=torch.load(path,weights_only=True);first=next(iter(payload['state']));payload['state'][first].fill_(float('nan'));torch.save(payload,path)
    base=model()
    with pytest.raises(ValueError,match='tensor'):load(base,path)
    assert not any(isinstance(layer,LoRALinear) for layer in base.modules())

@pytest.mark.parametrize('rank',[0,-1,True,100])
def test_invalid_rank_rejected(rank):
    with pytest.raises(ValueError):LoRALinear(torch.nn.Linear(8,8),rank=rank)

def test_double_attach_rejected():
    with pytest.raises(ValueError):attach(attach(model()))
