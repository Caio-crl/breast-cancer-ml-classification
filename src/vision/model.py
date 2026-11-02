"""MobileNetV3 Small binária, sem downloads implícitos ou treinamento."""

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MassMobileNet(nn.Module):
    """Duas saídas em logits; features congeladas mantêm BatchNorm em avaliação."""

    def __init__(self, backbone, freeze_features=False, trainable_feature_blocks=0):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, 2)
        self.configure_fine_tuning(trainable_feature_blocks if freeze_features else len(self.features))

    def configure_fine_tuning(self, trainable_feature_blocks=0):
        """Libere explicitamente os últimos N blocos; zero congela as features."""
        if (not isinstance(trainable_feature_blocks, int)
                or not 0 <= trainable_feature_blocks <= len(self.features)):
            raise ValueError("Quantidade inválida de blocos para fine-tuning.")
        cutoff = len(self.features) - trainable_feature_blocks
        for index, block in enumerate(self.features):
            block.requires_grad_(index >= cutoff)
        self.classifier.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for block in self.features:
            if not any(parameter.requires_grad for parameter in block.parameters()):
                block.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("Entrada esperada: (batch, 3, 224, 224).")
        features = self.features(images)
        return self.classifier(torch.flatten(self.avgpool(features), 1))


def resolve_pretrained_weights(weights_path=None):
    """Localize pesos locais sem iniciar download; resolução compartilhada com treino."""
    weights = Path(weights_path) if weights_path is not None else (
        Path(torch.hub.get_dir()) / "checkpoints" /
        Path(urlparse(MobileNet_V3_Small_Weights.IMAGENET1K_V1.url).path).name)
    if not weights.is_file():
        raise FileNotFoundError("Pesos ImageNet locais ausentes. Forneça weights_path ou "
                                "prepare o cache explicitamente; nenhum download foi iniciado.")
    return weights.resolve()


def build_model(pretrained=False, weights_path=None, freeze_features=None, trainable_feature_blocks=0):
    """Use pesos ImageNet somente de arquivo local ou cache já existente.

    pretrained=False não baixa nem procura pesos. Com pretrained=True, carregue
    o state_dict original de 1000 classes ANTES de trocar a camada final. Um
    arquivo ausente gera erro; não há fallback aleatório nem download automático.
    """
    if weights_path is not None and not pretrained:
        raise ValueError("weights_path requer pretrained=True.")
    weights = resolve_pretrained_weights(weights_path) if pretrained else None
    backbone = mobilenet_v3_small(weights=None)
    if weights is not None:
        backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    frozen = pretrained if freeze_features is None else freeze_features
    return MassMobileNet(backbone, freeze_features=frozen,
                         trainable_feature_blocks=trainable_feature_blocks)


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
