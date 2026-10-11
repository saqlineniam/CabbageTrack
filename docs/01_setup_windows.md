# 1. Setup (Windows 11)

The paper's results were produced on Windows 11 with Python 3.12.10 and an NVIDIA GeForce RTX 4060 (8 GB).
Linux and macOS work the same way (use `source .venv/bin/activate` and `/` in paths).

## 1.1 Install

1. **Python 3.12** from <https://www.python.org/downloads/> (tick "Add python.exe to PATH").
2. **Git** from <https://git-scm.com/download/win>.
3. **NVIDIA driver** (GPU only): a current Game Ready / Studio driver. A separate CUDA toolkit is not needed;
   PyTorch brings its own CUDA runtime.

## 1.2 Create the environment

Open *Command Prompt* in the folder where the project should live:

```bat
git clone https://github.com/saqlineniam/CabbageTrack.git
cd CabbageTrack
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch for your hardware (pick the command on <https://pytorch.org/get-started/locally/>). We used
torch 2.14.1 with CUDA 12.6:

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Without an NVIDIA GPU, `pip install torch torchvision` installs the CPU build: stages 2-3, the mosaic, the GNSS
comparison and all scoring run at the same speed; stage 1 (detection) and the demo video are much slower.

Then the rest, at the versions used for the paper, and the package itself:

```bat
pip install -r requirements.txt
pip install -e .
```

## 1.3 Check

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
cabbagetrack -h
python -m pytest tests -q
```

`torch.cuda.is_available()` should print `True` on a GPU machine. The tests run the pipeline on a short piece of a
flight and need the data record (next step); without it they are skipped.

## 1.4 Every new terminal

```bat
cd CabbageTrack
.venv\Scripts\activate
```

## Notes

* Paths with spaces need quotes: `--video "D:\My flights\flight 1.mp4"`.
* Long videos: stage 1 reads the video once; keep the videos on a local disk, not a network drive.
* If `pip install lap` fails, install the "Microsoft C++ Build Tools" or use `pip install lapx` (same module).
