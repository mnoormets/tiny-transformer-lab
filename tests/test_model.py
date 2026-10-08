import torch
import pytest
from model.gpt import Config,TinyGPT
from model.train import batch

def small():
    torch.manual_seed(7);torch.set_num_threads(2)
    return TinyGPT(Config(vocab_size=8,context=8,width=16,heads=2,layers=1))

def test_causal_mask_prevents_future_token_leakage():
    model=small().eval();a=torch.tensor([[1,2,3,4,5]]);b=torch.tensor([[1,2,3,7,0]])
    with torch.no_grad():
        left,_=model(a);right,_=model(b)
    torch.testing.assert_close(left[:,:3],right[:,:3],atol=1e-6,rtol=1e-6)
    assert not torch.allclose(left[:,3:],right[:,3:])

def test_next_token_batch_shift_and_boundaries():
    data=torch.arange(30);x,y=batch(data,8,4,torch.Generator().manual_seed(1))
    assert x.shape==y.shape==(4,8)
    torch.testing.assert_close(y,x+1)
    with pytest.raises(ValueError):batch(torch.arange(8),8,1,torch.Generator())

def test_real_gradient_updates_reduce_fixed_batch_loss():
    model=small();x=torch.tensor([[1,2,3,4,1,2,3,4]]);y=torch.tensor([[2,3,4,1,2,3,4,1]])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    _,before=model(x,y)
    for _ in range(20):
        _,loss=model(x,y);optimizer.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    _,after=model(x,y);assert after.item()<before.item()/2

def test_checkpoint_round_trip(tmp_path):
    model=small().eval();path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
    restored=small().eval();restored.load_state_dict(torch.load(path,weights_only=True))
    x=torch.tensor([[1,2,3]])
    torch.testing.assert_close(model(x)[0],restored(x)[0])

def test_shape_validation_and_generation():
    model=small();assert model(torch.tensor([[1,2,3]]))[0].shape==(1,3,8)
    with pytest.raises(ValueError):Config(vocab_size=8,width=15,heads=2)
    with pytest.raises(ValueError):model(torch.zeros(1,9,dtype=torch.long))
    assert model.generate(torch.tensor([[1]]),3).shape==(1,4)
