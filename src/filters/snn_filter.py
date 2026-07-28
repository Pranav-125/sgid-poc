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


