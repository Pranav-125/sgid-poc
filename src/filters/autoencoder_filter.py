import numpy as np
import torch 
import torch.nn as nn
from collections import deque
from src.filters.base_filter import BaseFliter
class SimpleAutoencoder(nn.Module):
    def  __init__(self,input_dim=5,bottleneck_dim=2):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,4),nn.ReLU(),nn.Linear(4,bottleneck_dim),nn.ReLU())
        self.decoder=nn.Sequential(nn.Linear(bottleneck_dim,4),nn.ReLU(),nn.Linear(4,input_dim))

    def forward(self,x):
        compressed=self.encoder(x)
        reconstructed=self.decoder(compressed)
        return reconstructed
    class AutoencoderFilter(BaseFilter):
        def __init__(self,window_size=5,threshold=None,epochs=50,retrain_every=200):
            self.window_size=window_size
            self.threshold=threshold
            self.epochs=epochs
            self.retrain_every=retrain_every
            self.model=SimpleAutoencoder(input_dim=window_size)
            self.buffer=deque(maxlen=window_size)
            self.values_since_retrain=0

        def _make_windows(self,data):
            windows=[]
            for i in range(len(data)-self.window_size+1):
                windows.append(data[i:i+self.window_size])
            return np.array(windows,dtype=np.float32)
        def fit(self,baseline_data):
            windows=self._make_windows(baseline_data)
            X=torch.tensor(windows)

            optimizer=torch.optim.Adam(self.model.paramters(),lr=0.01)
            loss_fn=nn.MSELoss()

            self.model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                reconstructed=self.model(X)
                loss=loss_fn(reconstructed,X)
                loss.backward()
                optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    reconstructed=self.model(X)
                    errors=torch.mean((reconstructed-X)**2,dim=1).numpy()
                self.threshold=float(np.mean(errors)+3*np.std(errors))

                for value in baseline_data[-self.window_size:]:
                    self.buffer.append(value)

        def detect(self,value,timestamp=None):
            self.buffer.append(value)
            if len(self.buffer)<self.window_size:
                return False,0.0

            window=np.array(self.buffer,dtype=np.float32)
            X=torch.tensor(window).unsqueeze(0)
            





        
    

