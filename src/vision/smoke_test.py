"""Um lote real, sem pesos, métricas, gradientes ou treinamento.

python -m src.vision.smoke_test --dataset-root data/vision/raw/cbis_ddsm
"""

import argparse
import importlib.util


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-dir", default="data/vision/processed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    if any(importlib.util.find_spec(name) is None for name in ("torch", "torchvision")):
        print("Smoke test não executado: torch/torchvision ausentes. Use um ambiente de visão separado.")
        return 2
    import torch
    from src.vision.dataset import build_dataloaders
    from src.vision.model import build_model, parameter_counts

    torch.manual_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(args.manifest_dir, args.dataset_root,
                                batch_size=args.batch_size, num_workers=0)
    images, labels = next(iter(loaders["train"]))
    model = build_model(pretrained=False).to(device).eval()
    with torch.inference_mode():
        logits = model(images.to(device))
    if tuple(images.shape[1:]) != (3, 224, 224) or tuple(logits.shape) != (len(labels), 2):
        raise RuntimeError("Formato inesperado no smoke test.")
    print(f"Dispositivo: {device}; imagens: {tuple(images.shape)}; logits: {tuple(logits.shape)}")
    print(f"Parâmetros: {parameter_counts(model)}; pesos pré-treinados: não")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
