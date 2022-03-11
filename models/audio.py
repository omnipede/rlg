import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np


class AudioEncoder:

    def __init__(self):

        pretrained_model = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
        self.model = Wav2Vec2Model.from_pretrained(pretrained_model)

    def encode(self, audio: np.ndarray):
        """
        Audio data structure loaded by librosa library
        :return: Embedding of audio
        """

        inputs = self.processor(audio, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.squeeze()
