import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class Wav2VecSequenceClassification(nn.Module):
    def __init__(self, wave2vec_model_name=None, hidden_size=768, num_classes=397):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wave2vec_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.wav2vec2(x).last_hidden_state
        x, _ = torch.max(x, dim=1)
        logit = self.classifier(x)
        return logit
