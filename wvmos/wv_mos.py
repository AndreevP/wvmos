from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch    
from collections import OrderedDict
import glob
import librosa
import tqdm
import numpy as np
from torch import nn

def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix):]] = weights[key]
    return result     


class Wav2Vec2ConvEncoder:

    def __init__(self, device="cuda"):
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").feature_extractor
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000
        self.device = device

    def __call__(self, x):
        # x - [bs, 1, time]
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device))
        return hidden_states
    
class Wav2Vec2FullEncoder:

    def __init__(self, device="cuda"):
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.preprocessor._sample_rate = 16000
        self.device = device

    def __call__(self, x):
        # x - [bs, 1, time]
        x = x[:, 0]
        input_values = (x - x.mean(-1)[:, None]) / (x.std(-1)[:, None] + 1e-6)
        hidden_states = self.encoder(input_values.to(self.device)).last_hidden_state
        return hidden_states.transpose(-2, -1)
    
    
class Wav2Vec2MOS(nn.Module):
    def __init__(self, path, freeze=True, cuda=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze
        
        self.dense = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(extract_prefix('model.', torch.load(path)['state_dict']))
        self.eval()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
    def forward(self, x):
        x = self.encoder(x)['last_hidden_state'] # [Batch, time, feats]
        x = self.dense(x) # [batch, time, 1]
        x = x.mean(dim=[1,2], keepdims=True) # [batch, 1, 1]
        return x
                
    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
            
    def calculate_dir(self, path, mean=True):
        
        pred_mos = []
        for path in tqdm.tqdm(sorted(glob.glob(f"{path}/*.wav"))):
            signal = librosa.load(path, sr=16_000)[0]
            x = self.processor(signal, return_tensors="pt", padding=True, sampling_rate=16000).input_values
            if self.cuda_flag:
                x = x.cuda()
            with torch.no_grad():
                res = self.forward(x).mean()
            pred_mos.append(res.item())
        if mean:
            return np.mean(pred_mos)
        else:
            return pred_mos
        
    def calculate_one(self, path):
        signal = librosa.load(path, sr=16_000)[0]
        x = self.processor(signal, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        with torch.no_grad():
            if self.cuda_flag:
                x = x.cuda()
            res = self.forward(x).mean()
        return res.cpu().item()