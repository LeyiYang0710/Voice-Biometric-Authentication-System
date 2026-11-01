"""Small TDNN: audio -> acoustic features -> embedding -> class scores.

Inspired by the SpeechBrain speaker_id teaching recipe; trained from scratch.
This is not CAM++ and does not establish reliable unknown-speaker rejection.
"""

# 1. Import neural network components and audio feature tools.
import torch
from torch import nn
import torchaudio


class SmallTDNN(nn.Module):
    # 2. Define layers; retain attribute names for checkpoint compatibility.
    def __init__(self, n_classes, embedding_dim=64, cosine_classifier=False):
        super().__init__()

        # Audio frontend: 16kHz input, 25ms window, 10ms hop, 40 Mel bands.
        self.frontend = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40,
            center=False,
            power=2.0,
        )

        # Each tuple: input channels, output channels, kernel size, dilation.
        layer_settings = [(40, 64, 5, 1), (64, 64, 3, 2), (64, 128, 3, 3)]
        blocks = []
        for input_channels, output_channels, kernel_size, dilation in layer_settings:
            blocks.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=dilation * (kernel_size - 1) // 2,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(),
                ]
            )
        self.tdnn = nn.Sequential(*blocks)

        # Project 128 means + 128 standard deviations into a 64-D embedding.
        self.projection = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.cosine_classifier = cosine_classifier
        self.classifier.bias.requires_grad_(not cosine_classifier)

    # 3. Extract an embedding: log-mel -> convolution -> pooling -> projection.
    def embed(self, wav):
        features = self.frontend(wav).clamp_min(1e-6).log()
        features = features - features.mean(dim=-1, keepdim=True)

        frame_features = self.tdnn(features)
        mean = frame_features.mean(dim=-1)
        standard_deviation = (
            frame_features.var(dim=-1, unbiased=False).clamp_min(1e-6).sqrt()
        )
        statistics = torch.cat([mean, standard_deviation], dim=1)
        return self.projection(statistics)

    # 4. During training, classify the embedding into a training speaker class.
    def forward(self, wav):
        embedding = self.embed(wav)
        if self.cosine_classifier:
            # Scaled cosine logits; no bias or angular margin is applied.
            embedding = nn.functional.normalize(embedding, dim=1)
            weights = nn.functional.normalize(self.classifier.weight, dim=1)
            return 16.0 * nn.functional.linear(embedding, weights)
        return self.classifier(embedding)
