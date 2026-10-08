import torch
from pathlib import Path
from .gpt import Config,TinyGPT

def main():
    checkpoint=torch.load(Path(__file__).resolve().parents[1]/'artifacts/checkpoint.pt',map_location='cpu',weights_only=True)
    model=TinyGPT(Config(**checkpoint['config']));model.load_state_dict(checkpoint['state_dict'])
    torch.manual_seed(42);torch.set_num_threads(2)
    chars=checkpoint['characters'];ids=model.generate(torch.tensor([[chars.index('T')]]),120)
    print(''.join(chars[i] for i in ids[0].tolist()))
if __name__=='__main__':main()
