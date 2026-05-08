# Reproduction Record

## Summary

This project targets a focused PDEBench reproduction path: **1D Burgers forecasting with a NeuralOperator Fourier Neural Operator**.

The current checked-in result is a real-data development verification run. It demonstrates that the data reader, NeuralOperator model, autoregressive rollout, metrics, checkpointing, and visualization all work together. It is not a final full-data benchmark score.

## Experiment Definition

- PDE: 1D Burgers
- Model: `neuralop.models.FNO`
- Input: first `10` solution frames
- Prediction target: future trajectory frames
- Loss: autoregressive one-step MSE accumulated across rollout steps
- Evaluation metrics: MSE, RMSE, relative L2 over predicted future frames
- Full-data config: `configs/burgers_fno_neuralop.yaml`
- Development config: `configs/burgers_fno_neuralop_development.yaml`

## Data

Full official data:

- File: `1D_Burgers_Sols_Nu0.01.hdf5`
- Size: 8.23GB
- Source: `https://huggingface.co/datasets/pdebench/Burgers/resolve/main/1D_Burgers_Sols_Nu0.01.hdf5`
- Original DOI: `https://doi.org/10.18419/darus-2986`

Development verification data:

- File: `1D_Burgers_Sols_Nu0.01_development.hdf5`
- Shape observed locally: `tensor=(50, 201, 1024)`, `x-coordinate=(1024,)`, `t-coordinate=(201,)`
- Used only for small end-to-end validation.

Data files are excluded from Git. The downloader supports `.part` resume for large downloads.

## Current Result

Development run command:

```powershell
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" scripts\download_pdebench.py --name burgers_nu001_development --output-dir data
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" run_burgers_fno.py --config configs\burgers_fno_neuralop_development.yaml
```

Metrics:

```json
{
  "mse": 0.193971399217844,
  "relative_l2": 0.9061306118965149,
  "rmse": 0.41577601432800293,
  "best_relative_l2": 0.9061306118965149
}
```

The relative L2 is high because this is a one-epoch development run over a small subset. It should be used as evidence of a working reproduction pipeline, not as a full reproduction result.

## Validation

Verified locally with:

```powershell
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" -m py_compile run_burgers_fno.py scripts\download_pdebench.py
& "C:\Project\anaconda3\envs\torch_cuda\python.exe" run_burgers_fno.py --config configs\smoke_synthetic_neuralop.yaml
```

Environment facts observed locally:

- Python environment: `torch_cuda`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- `torch.cuda.is_available() == True`
- Installed and verified: `h5py`, `neuralop`

## Next Steps

- Complete the 8.23GB full-data download from the Hugging Face PDEBench mirror.
- Run `configs/burgers_fno_neuralop.yaml` for a longer training schedule.
- Add a compact comparison table against PDEBench baseline settings once full-data metrics are available.
