# AGENTS

## Python environment

- Use a local virtual environment for project dependencies.
- Keep Python and package versions aligned with `pyproject.toml` and `uv.lock`.
- Activate the environment before running project-specific commands if you are not using `uv run`.

## NVIDIA and CUDA guidance

- Ensure the NVIDIA driver is installed and the GPU is visible from the current system before running ML or TTS workloads.
- Install PyTorch builds that match the CUDA runtime supported by your driver and the wheel you are using.
- You usually do not need the full CUDA toolkit unless you are compiling custom CUDA extensions.

## Validation commands

GPU visibility:

```bash
nvidia-smi
```

Python/CUDA validation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## UV workflow

- Use `pyproject.toml` as the single source of truth for dependencies and project metadata.
- Generate and commit `uv.lock` for reproducible environments.
- Create or update the environment with `uv sync`.
- Add packages with `uv add <package>` and development tools with `uv add --dev <package>`.
- After dependency changes, run `uv lock` and commit both `pyproject.toml` and `uv.lock`.

Recommended command flow:

```bash
uv venv
uv sync
```

## Project run commands

- Preferred app start command:

```bash
uv run espeech_gradio.py
```

- `uv run espeech_gradio.py` is equivalent to `uv run python espeech_gradio.py` for `.py` files.
- If dependencies are not synced yet, run:

```bash
uv sync
uv run espeech_gradio.py
```

- The app entrypoint is `espeech_gradio.py`; it starts Gradio via `app.launch()`.
- On first run, the app may download model artifacts from Hugging Face repo `ESpeech/ESpeech-TTS-1_RL-V2`.
- If authenticated download is required, set:

```bash
export HF_TOKEN=your_token
```

Dependency updates:

```bash
uv add <package>
uv add --dev pytest ruff
uv lock
uv sync
```
