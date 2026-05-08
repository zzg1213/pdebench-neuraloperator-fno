from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Full PDEBench reproduction requires h5py. Install it outside torch_cuda "
            "or ask for approval before modifying the torch_cuda environment."
        ) from exc
    return h5py


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PDEBenchScalarDataset(Dataset):
    """Loads scalar PDEBench HDF5 files shaped as [sample, time, x]."""

    def __init__(
        self,
        path: Path,
        initial_step: int,
        reduced_resolution: int = 1,
        reduced_resolution_t: int = 1,
        reduced_batch: int = 1,
        max_samples: int = -1,
    ) -> None:
        h5py = require_h5py()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        with h5py.File(path, "r") as handle:
            if "tensor" not in handle:
                raise KeyError(f"{path} does not contain the PDEBench scalar key 'tensor'.")
            data = np.asarray(handle["tensor"], dtype=np.float32)
            grid = np.asarray(handle["x-coordinate"], dtype=np.float32)

        if data.ndim == 3:
            data = data[..., None]
        if data.ndim != 4:
            raise ValueError(f"Expected PDEBench tensor shape [batch, time, x, channel], got {data.shape}.")
        data = data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, :]
        if max_samples > 0:
            data = data[:max_samples]
        data = np.transpose(data, (0, 2, 1, 3))
        self.data = torch.from_numpy(data)
        self.grid = torch.from_numpy(grid[::reduced_resolution]).float().unsqueeze(-1)
        self.initial_step = initial_step

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.data[index]
        x = y[..., : self.initial_step, :]
        return x, y, self.grid


class SyntheticBurgersLikeDataset(Dataset):
    def __init__(self, samples: int, spatial_points: int, timesteps: int, initial_step: int) -> None:
        x = torch.linspace(0, 1, spatial_points)
        rows = []
        for sample in range(samples):
            phase = 2.0 * math.pi * sample / max(samples, 1)
            freq = 1 + sample % 3
            trajectory = []
            for t_idx in range(timesteps):
                t = t_idx / max(timesteps - 1, 1)
                wave = torch.sin(2.0 * math.pi * freq * (x - 0.2 * t) + phase)
                decay = torch.exp(torch.tensor(-0.4 * t))
                trajectory.append(decay * wave)
            rows.append(torch.stack(trajectory, dim=-1))
        self.data = torch.stack(rows, dim=0).unsqueeze(-1)
        self.grid = x.unsqueeze(-1)
        self.initial_step = initial_step

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.data[index]
        x = y[..., : self.initial_step, :]
        return x, y, self.grid


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class TorchFNO1d(nn.Module):
    """Small local FNO used only for smoke-testing when NeuralOperator is absent."""

    def __init__(self, in_channels: int, out_channels: int, modes: int, width: int, layers: int) -> None:
        super().__init__()
        self.lift = nn.Conv1d(in_channels + 1, width, kernel_size=1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        grid_ch = grid.transpose(1, 2)
        x = torch.cat([x, grid_ch], dim=1)
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = torch.nn.functional.gelu(spectral(x) + local(x))
        return self.project(x)


class NeuralOperatorFNO1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, width: int, layers: int) -> None:
        super().__init__()
        try:
            from neuralop.models import FNO
        except ImportError as exc:
            raise RuntimeError(
                "backend=neuralop requires the neuraloperator package. Install it outside "
                "torch_cuda or ask for approval before modifying the torch_cuda environment."
            ) from exc
        self.model = FNO(
            n_modes=(modes,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=width,
            n_layers=layers,
            positional_embedding="grid",
        )

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        del grid
        return self.model(x)


@dataclass
class Metrics:
    mse: float
    relative_l2: float
    rmse: float


def channel_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 2, 1).reshape(x.shape[0], -1, x.shape[1])


def one_step_target(y: torch.Tensor, t: int) -> torch.Tensor:
    return y[..., t : t + 1, :].permute(0, 3, 2, 1).squeeze(2)


def autoregressive_predict(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    grid: torch.Tensor,
    initial_step: int,
    t_train: int,
) -> torch.Tensor:
    window = x
    pred = [y[..., step : step + 1, :] for step in range(initial_step)]
    for t in range(initial_step, t_train):
        out = model(channel_first(window), grid).permute(0, 2, 1).unsqueeze(-2)
        pred.append(out)
        window = torch.cat([window[..., 1:, :], out], dim=-2)
    return torch.cat(pred, dim=-2)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, initial_step: int) -> Metrics:
    pred_eval = pred[..., initial_step:, :]
    target_eval = target[..., initial_step : pred.shape[-2], :]
    mse = torch.mean((pred_eval - target_eval) ** 2)
    rmse = torch.sqrt(mse)
    relative_l2 = torch.linalg.vector_norm(pred_eval - target_eval) / torch.linalg.vector_norm(target_eval)
    return Metrics(mse=float(mse), relative_l2=float(relative_l2), rmse=float(rmse))


def make_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, int, int]:
    data_config = config["data"]
    if data_config["kind"] == "pdebench_hdf5":
        dataset = PDEBenchScalarDataset(
            path=Path(data_config["path"]),
            initial_step=data_config["initial_step"],
            reduced_resolution=data_config.get("reduced_resolution", 1),
            reduced_resolution_t=data_config.get("reduced_resolution_t", 1),
            reduced_batch=data_config.get("reduced_batch", 1),
            max_samples=data_config.get("max_samples", -1),
        )
    elif data_config["kind"] == "synthetic_burgers_like":
        dataset = SyntheticBurgersLikeDataset(
            samples=data_config["samples"],
            spatial_points=data_config["spatial_points"],
            timesteps=data_config["timesteps"],
            initial_step=data_config["initial_step"],
        )
    else:
        raise ValueError(f"Unknown data.kind: {data_config['kind']}")

    train_size = int(len(dataset) * data_config["train_ratio"])
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config["seed"])
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_data,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, dataset.data.shape[1], dataset.data.shape[-1]


def build_model(config: dict[str, Any], in_channels: int, out_channels: int) -> nn.Module:
    model_config = config["model"]
    common = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "modes": model_config["modes"],
        "width": model_config["width"],
        "layers": model_config["layers"],
    }
    if model_config["backend"] == "neuralop":
        return NeuralOperatorFNO1d(**common)
    if model_config["backend"] == "torch_fno":
        return TorchFNO1d(**common)
    raise ValueError(f"Unknown model.backend: {model_config['backend']}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    initial_step: int,
    t_train: int,
) -> tuple[Metrics, tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    total_mse = 0.0
    total_relative_l2 = 0.0
    total_rmse = 0.0
    batches = 0
    sample = None
    with torch.no_grad():
        for x, y, grid in loader:
            x = x.to(device)
            y = y.to(device)
            grid = grid.to(device)
            t_cap = min(t_train, y.shape[-2])
            pred = autoregressive_predict(model, x, y, grid, initial_step, t_cap)
            metrics = batch_metrics(pred, y, initial_step)
            total_mse += metrics.mse
            total_relative_l2 += metrics.relative_l2
            total_rmse += metrics.rmse
            batches += 1
            if sample is None:
                sample = (pred[0].detach().cpu(), y[0, :, :t_cap].detach().cpu())
    return (
        Metrics(
            mse=total_mse / batches,
            relative_l2=total_relative_l2 / batches,
            rmse=total_rmse / batches,
        ),
        sample,
    )


def plot_prediction(sample: tuple[torch.Tensor, torch.Tensor] | None, output_path: Path) -> None:
    if sample is None:
        return

    def heatmap(image: np.ndarray) -> Image.Image:
        image = image.astype(np.float32)
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if max_value - min_value < 1e-12:
            scaled = np.zeros_like(image, dtype=np.float32)
        else:
            scaled = (image - min_value) / (max_value - min_value)
        red = np.clip(1.5 * scaled, 0.0, 1.0)
        green = np.clip(1.5 - np.abs(2.0 * scaled - 1.0), 0.0, 1.0)
        blue = np.clip(1.5 * (1.0 - scaled), 0.0, 1.0)
        rgb = np.stack([red, green, blue], axis=-1)
        return Image.fromarray((255 * rgb).astype(np.uint8)).resize((240, 180), Image.Resampling.BILINEAR)

    pred, target = sample
    pred_2d = pred[..., 0].transpose(0, 1).numpy()
    target_2d = target[..., 0].transpose(0, 1).numpy()
    panels = [
        ("target", heatmap(target_2d)),
        ("prediction", heatmap(pred_2d)),
        ("absolute error", heatmap(np.abs(pred_2d - target_2d))),
    ]
    gutter = 8
    title_h = 24
    width = sum(panel.width for _, panel in panels) + gutter * (len(panels) + 1)
    height = title_h + panels[0][1].height + gutter * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x_offset = gutter
    for title, panel in panels:
        draw.text((x_offset, 4), title, fill=(0, 0, 0))
        canvas.paste(panel, (x_offset, title_h + gutter))
        x_offset += panel.width + gutter
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def train(config: dict[str, Any]) -> dict[str, float]:
    set_seed(config["seed"])
    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = Path(config["output"]["root"]) / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_loader, val_loader, _, num_channels = make_dataloaders(config)
    initial_step = config["data"]["initial_step"]
    t_train = config["data"]["t_train"]
    model = build_model(config, in_channels=initial_step * num_channels, out_channels=num_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"]["scheduler_step"],
        gamma=config["train"]["scheduler_gamma"],
    )
    loss_fn = nn.MSELoss()
    best_relative_l2 = float("inf")
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for x, y, grid in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            grid = grid.to(device)
            t_cap = min(t_train, y.shape[-2])
            window = x
            loss = torch.zeros((), device=device)
            for t in range(initial_step, t_cap):
                out = model(channel_first(window), grid).permute(0, 2, 1).unsqueeze(-2)
                target = y[..., t : t + 1, :]
                loss = loss + loss_fn(out, target)
                window = torch.cat([window[..., 1:, :], out], dim=-2)
            loss = loss / max(t_cap - initial_step, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["train"].get("grad_clip_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        row = {"epoch": epoch, "train_mse": train_loss / batches}
        if epoch % config["train"]["eval_interval"] == 0:
            val_metrics, sample = evaluate(model, val_loader, device, initial_step, t_train)
            row.update(
                {
                    "val_mse": val_metrics.mse,
                    "val_relative_l2": val_metrics.relative_l2,
                    "val_rmse": val_metrics.rmse,
                }
            )
            if val_metrics.relative_l2 < best_relative_l2:
                best_relative_l2 = val_metrics.relative_l2
                torch.save({"model": model.state_dict(), "config": config, "metrics": row}, output_dir / "best.pt")
                if config["output"].get("plot", True):
                    plot_prediction(sample, output_dir / "prediction.png")
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    final_metrics, sample = evaluate(model, val_loader, device, initial_step, t_train)
    result = {
        "mse": final_metrics.mse,
        "relative_l2": final_metrics.relative_l2,
        "rmse": final_metrics.rmse,
        "best_relative_l2": best_relative_l2,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"final": result, "history": history}, handle, indent=2)
    if config["output"].get("plot", True):
        plot_prediction(sample, output_dir / "prediction.png")
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PDEBench Burgers FNO with NeuralOperator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--copy-config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.copy_config:
        shutil.copyfile(args.config, args.copy_config)
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
