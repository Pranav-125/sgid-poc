"""
snn_filter.py

Purpose: Detects anomalies using a Spiking Neural Network (SNN).
Converts incoming metric values into spike trains, feeds them through
a small network of Leaky Integrate-and-Fire (LIF) neurons, and treats
a strong output spike as an anomaly signal.
Implements the BaseFilter interface so it's swappable with the others.
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from collections import deque
from src.filters.base_filter import BaseFilter
class SpikingNet(nn.Module):
   
    def __init__(self, input_dim=5, hidden_dim=8, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_dim, 1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps=10):
        """
        x: a single window of spike-encoded input, shape (input_dim,)
        Runs the network over `num_steps` internal timesteps,
        accumulating membrane potential, and returns the total
        number of output spikes fired (used as the anomaly score).
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spike_count = 0
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_count += spk2.item()

        return spike_count


class SNNFilter(BaseFilter):
    def __init__(self, window_size=5, threshold=None, epochs=50,
                 num_steps=10, retrain_every=200):
        """
        window_size:   how many recent values form one input window
        threshold:     spike-count above this = anomaly (auto-set during fit)
        epochs:        training iterations on baseline data
        num_steps:     internal simulation timesteps per detection call
        retrain_every: how often to refit on recent data
        """
        self.window_size = window_size
        self.threshold = threshold
        self.epochs = epochs
        self.num_steps = num_steps
        self.retrain_every = retrain_every

        self.model = SpikingNet(input_dim=window_size)
        self.buffer = deque(maxlen=window_size)
        self.values_since_retrain = 0

        # Used to normalize raw metric values into spike rates (0 to 1)
        self.data_min = 0.0
        self.data_max = 100.0

    def _encode(self, window):
        """
        Rate-encoding: normalize raw values (e.g. CPU%) into a 0-1 range,
        which snntorch interprets as spike probability/intensity per step.
        """
        arr = np.array(window, dtype=np.float32)
        arr = np.clip(arr, self.data_min, self.data_max)
        normalized = (arr - self.data_min) / (self.data_max - self.data_min + 1e-8)
        return torch.tensor(normalized)

    def _make_windows(self, data):
        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])
        return windows

    def fit(self, baseline_data):
        """
        Trains the SNN so that normal windows produce LOW spike counts,
        establishing a baseline "quiet" firing rate. Then sets the
        anomaly threshold from the spread of spike counts on that
        known-normal data.
        """
        self.data_min = min(baseline_data)
        self.data_max = max(baseline_data)

        windows = self._make_windows(baseline_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for window in windows:
                x = self._encode(window)
                optimizer.zero_grad()

                spike_count = self._forward_differentiable(x)
                # Target: as close to zero spikes as possible for normal data
                target = torch.tensor(0.0)
                loss = loss_fn(spike_count, target)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Determine threshold from baseline spike counts
        self.model.eval()
        baseline_scores = []
        with torch.no_grad():
            for window in windows:
                x = self._encode(window)
                score = self.model(x, num_steps=self.num_steps)
                baseline_scores.append(score)

        self.threshold = float(np.mean(baseline_scores) + 3 * np.std(baseline_scores)) or 1.0

        for value in baseline_data[-self.window_size:]:
            self.buffer.append(value)

    def _forward_differentiable(self, x):
        """
        Same as SpikingNet.forward, but returns a tensor (not .item())
        so gradients can flow through it during training.
        """
        mem1 = self.model.lif1.init_leaky()
        mem2 = self.model.lif2.init_leaky()
        spike_sum = torch.tensor(0.0)

        for _ in range(self.num_steps):
            cur1 = self.model.fc1(x)
            spk1, mem1 = self.model.lif1(cur1, mem1)
            cur2 = self.model.fc2(spk1)
            spk2, mem2 = self.model.lif2(cur2, mem2)
            spike_sum = spike_sum + spk2.squeeze()

        return spike_sum

    def detect(self, value, timestamp=None):
        """
        Adds new value to the rolling window, encodes it into spikes,
        runs it through the SNN, and checks if the output spike count
        crosses the anomaly threshold.
        Returns (should_escalate: bool, score: float)
        """
        self.buffer.append(value)

        if len(self.buffer) < self.window_size or self.threshold is None:
            return False, 0.0

        x = self._encode(list(self.buffer))

        self.model.eval()
        with torch.no_grad():
            spike_score = self.model(x, num_steps=self.num_steps)

        should_escalate = spike_score > self.threshold

        self.values_since_retrain += 1
        if self.values_since_retrain >= self.retrain_every:
            self.fit(list(self.buffer))
            self.values_since_retrain = 0

        return should_escalate, float(spike_score)

    def name(self):
        return "snn"


