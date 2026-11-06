"""Small-model training: prepare data -> train -> validate -> save weights.

train initializes a small TDNN; evaluate and embed load saved weights.
Start reading at main(). Formal testing requires an explicit --split test.
"""

# 1. Import tools and locate the data manifests.
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from small_model import SmallTDNN

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/manifests/small_model"
SAMPLE_RATE = 16000
CROP_SAMPLES = 48000  # Use 3 seconds of audio per input.


# 2. Dataset: use random 3-second training crops and centered validation crops.
class Recordings(Dataset):
    def __init__(self, split, training=False):
        manifest_path = MANIFEST / (split + ".json")
        self.rows = json.loads(manifest_path.read_text())
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        recording = self.rows[index]
        recording_path = ROOT / recording["path"]
        information = sf.info(recording_path)

        if self.training:
            start = random.randint(0, information.frames - CROP_SAMPLES)
        else:
            start = (information.frames - CROP_SAMPLES) // 2

        audio, sample_rate = sf.read(
            recording_path, start=start, frames=CROP_SAMPLES, dtype="float32"
        )
        if (
            sample_rate != SAMPLE_RATE
            or len(audio) != CROP_SAMPLES
            or not np.isfinite(audio).all()
        ):
            raise ValueError(str(recording_path))

        return torch.from_numpy(audio), recording["label"]


# 3. Read command-line options for training, input files, and output paths.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Small TDNN training and embedding extraction"
    )
    parser.add_argument("command", choices=["train", "evaluate", "embed"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3601)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--cosine-classifier", action="store_true",
        help="Train with normalized embeddings/weights and fixed logit scale 16",
    )
    parser.add_argument(
        "--margin", type=float, default=0.0,
        help="Training-only additive cosine margin; requires --cosine-classifier",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/small_tdnn_v1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--vector-output", type=Path)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-valid-batches", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.margin < 1:
        parser.error("--margin must be in [0, 1)")
    if args.command == "train" and args.margin and not args.cosine_classifier:
        parser.error("--margin requires --cosine-classifier during training")
    return parser, args


# 4. Set random seeds and check the requested device for repeatable experiments.
def configure_environment(args):
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False


# 5. Validate: compute loss and correct predictions without updating weights.
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (audio, targets) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break

            audio = audio.to(device)
            targets = targets.to(device)
            logits = model(audio)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(targets)

    if total == 0:
        raise ValueError("No evaluation examples")
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


# 6. Train one epoch: predict -> compute loss -> backpropagate -> update.
def train_one_epoch(model, loader, optimizer, args, epoch, step):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for audio, targets in loader:
        audio = audio.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(audio)
        loss_logits = logits
        if args.margin:
            # AM-Softmax: subtract margin before the fixed cosine scale of 16.
            loss_logits = logits.clone()
            loss_logits[torch.arange(len(targets), device=targets.device), targets] -= (
                16.0 * args.margin
            )
        loss = torch.nn.functional.cross_entropy(loss_logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")

        loss.backward()
        optimizer.step()
        step += 1

        loss_sum += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        total += len(targets)

        if step % 20 == 0 or args.max_steps:
            print(f"epoch={epoch} step={step} loss={loss.item():.4f}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    return step, loss_sum / total, correct / total


# 7. Record parameters, versions, data splits, and source checksums.
def build_configuration(args, model):
    configuration = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    configuration.update(
        torch=str(torch.__version__),
        python=platform.python_version(),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        initialization="random; all trainable network parameters updated",
        audio="3 seconds: random crop during training, center crop during validation/inference; mono 16kHz",
        features="40 log-mel bands, 25ms window, 10ms hop, per-crop temporal mean subtraction; see small_model.py",
        smoke_only=bool(args.max_steps or args.max_valid_batches),
        classifier=(
            f"scaled cosine, scale=16, training margin={args.margin}"
            if model.cosine_classifier else "linear"
        ),
        manifest_summary=json.loads((MANIFEST / "summary.json").read_text()),
        code_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["train_small.py", "small_model.py", "prepare_small_data.py"]
        },
    )
    return configuration


# 8. Verify identical reloaded outputs and an actual convolution weight update.
def verify_checkpoint(model, valid_loader, labels, initial_weight, args, steps, config):
    sample = next(iter(valid_loader))[0][:2].to(args.device)
    model.eval()
    restored_model = SmallTDNN(
        len(labels), cosine_classifier=model.cosine_classifier
    ).to(args.device)
    checkpoint = torch.load(
        args.output / "last.pt", map_location=args.device, weights_only=True
    )
    restored_model.load_state_dict(checkpoint["model"])
    restored_model.eval()

    with torch.no_grad():
        torch.testing.assert_close(
            model(sample), restored_model(sample), rtol=0, atol=0
        )

    weights_changed = not torch.equal(initial_weight, model.tdnn[0].weight.detach())
    if not weights_changed:
        raise RuntimeError("Backbone weights did not change")

    verification = {
        "checkpoint_reload_equal": True,
        "backbone_weights_changed": weights_changed,
        "finite_training_loss": True,
        "steps": steps,
        "test_set_used": False,
        "smoke_only": config["smoke_only"],
    }
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
    print("Checkpoint reload verified; final test was NOT run.", flush=True)


# 9. Train and validate each epoch; select weights using validation loss only.
def run_training(args, model, labels):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists():
        raise ValueError("Output contains a prior run; choose a new --output")

    config = build_configuration(args, model)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    train_loader = DataLoader(
        Recordings("train", training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        Recordings("valid"), batch_size=args.batch_size, num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_loss = float("inf")
    step = 0
    start_time = time.perf_counter()
    initial_weight = model.tdnn[0].weight.detach().clone()

    for epoch in range(1, args.epochs + 1):
        step, train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, args, epoch, step
        )
        validation = evaluate(model, valid_loader, args.device, args.max_valid_batches)
        record = {
            "epoch": epoch,
            "steps": step,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "labels": labels,
            "epoch": epoch,
            "config": config,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, args.output / "best.pt")

        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        if args.max_steps and step >= args.max_steps:
            break

    verify_checkpoint(model, valid_loader, labels, initial_weight, args, step, config)


# 10. Load saved weights and verify the speaker label mapping.
def load_checkpoint(args, model, labels):
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    if checkpoint["labels"] != labels:
        raise ValueError("Label map mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    # Old checkpoints used the linear classifier and have no cosine flag.
    model.cosine_classifier = checkpoint["config"].get("cosine_classifier", False)
    model.classifier.bias.requires_grad_(not model.cosine_classifier)
    model.eval()
    return checkpoint


# 11. Evaluate a saved model using public-data closed-set classification.
def run_evaluation(args, model, checkpoint):
    loader = DataLoader(
        Recordings(args.split), batch_size=args.batch_size, num_workers=0
    )
    result = evaluate(model, loader, args.device)
    result.update(
        split=args.split,
        checkpoint=str(args.checkpoint),
        smoke_only=checkpoint["config"]["smoke_only"],
        protocol="Closed-set public speaker classification; not member/Unknown accuracy",
    )
    result_path = args.checkpoint.parent / ("evaluation-" + args.split + ".json")
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# 12. Extract a 64-D embedding from a recording using saved weights.
def run_embedding(args, model, checkpoint):
    audio, sample_rate = sf.read(args.wav, dtype="float32")
    if (
        sample_rate != SAMPLE_RATE
        or audio.ndim != 1
        or len(audio) < CROP_SAMPLES
        or not np.isfinite(audio).all()
        or np.max(np.abs(audio)) == 0
    ):
        raise ValueError("Need non-silent mono 16kHz audio >=3 seconds")

    start = (len(audio) - CROP_SAMPLES) // 2
    audio_crop = torch.from_numpy(audio[start : start + CROP_SAMPLES])
    audio_batch = audio_crop.unsqueeze(0).to(args.device)

    with torch.no_grad():
        embedding = model.embed(audio_batch)
        normalized = torch.nn.functional.normalize(embedding, dim=1)
        vector = normalized[0].cpu().numpy()

    if args.vector_output:
        args.vector_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.vector_output, vector)

    result = {
        "dimension": len(vector),
        "finite": bool(np.isfinite(vector).all()),
        "norm": float(np.linalg.norm(vector)),
        "smoke_only": checkpoint["config"]["smoke_only"],
        "note": "64-dim small-model embedding; independent registration and threshold calibration required",
    }
    print(json.dumps(result, indent=2))


# 13. Entry point: build the model, then train, evaluate, or extract embeddings.
def main():
    parser, args = parse_arguments()
    configure_environment(args)
    labels = json.loads((MANIFEST / "labels.json").read_text())
    model = SmallTDNN(
        len(labels), cosine_classifier=args.cosine_classifier
    ).to(args.device)

    if args.command == "train":
        run_training(args, model, labels)
        return

    if not args.checkpoint:
        parser.error("--checkpoint required")
    checkpoint = load_checkpoint(args, model, labels)

    if args.command == "evaluate":
        run_evaluation(args, model, checkpoint)
    else:
        if not args.wav:
            parser.error("--wav required")
        run_embedding(args, model, checkpoint)


if __name__ == "__main__":
    main()
