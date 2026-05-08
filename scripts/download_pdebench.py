from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm


DATASETS = {
    "burgers_nu001": {
        "filename": "1D_Burgers_Sols_Nu0.01.hdf5",
        "url": "https://darus.uni-stuttgart.de/api/access/datafile/133136",
        "size": 8232968312,
    },
    "burgers_nu001_hf": {
        "filename": "1D_Burgers_Sols_Nu0.01.hdf5",
        "url": "https://huggingface.co/datasets/pdebench/Burgers/resolve/main/1D_Burgers_Sols_Nu0.01.hdf5",
        "size": 8232968312,
    },
    "burgers_nu001_development": {
        "filename": "1D_Burgers_Sols_Nu0.01_development.hdf5",
        "url": "https://huggingface.co/datasets/LDA1020/codepde-data/resolve/main/burgers/1D_Burgers_Sols_Nu0.01_development.hdf5",
        "size": 41171748,
    },
}


def download(url: str, destination: Path, expected_size: int | None = None, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size != expected_size:
        if partial.exists():
            partial.unlink()
        destination.replace(partial)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        print(f"Complete: {destination}")
        return

    resume_at = 0 if not partial.exists() else partial.stat().st_size
    headers = {"Range": f"bytes={resume_at}-"} if resume_at > 0 else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if resume_at > 0 and response.status_code != 206:
            resume_at = 0
        content_length = int(response.headers.get("content-length", 0))
        total = resume_at + content_length if resume_at > 0 else content_length
        mode = "ab" if resume_at > 0 else "wb"
        with partial.open(mode) as handle:
            with tqdm(
                total=total,
                initial=resume_at,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))
    if expected_size and partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download: {partial.stat().st_size} / {expected_size} bytes in {partial}")
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected PDEBench files.")
    parser.add_argument("--name", choices=sorted(DATASETS), default="burgers_nu001")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    spec = DATASETS[args.name]
    destination = args.output_dir / spec["filename"]
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and not args.force:
        print(f"Resuming: {partial} ({partial.stat().st_size} bytes)")
    download(spec["url"], destination, expected_size=spec.get("size"), force=args.force)
    print(f"Downloaded: {destination}")


if __name__ == "__main__":
    main()
