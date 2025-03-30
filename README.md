# TripoSG

The [TripoSG](https://github.com/VAST-AI-Research/TripoSG/) project,
packaged with uv for easier installation.

## Requirements

- The `TripoSG` Python library can be run on Python 3.12
- The Gradio app (`spaces`) depends on
  - cvcuda-cu12 which only supports up to 3.11
  - TRELLIS HF Space's nvdiffrast wheel
    ([here](https://huggingface.co/spaces/JeffreyXiang/TRELLIS/tree/main/wheels))
    which only supports up to 3.10 -- REPLACED by [diffrp-nvdiffrast](https://pypi.org/project/diffrp-nvdiffrast)

## Usage

### Library

To run just the library:

```bash
uv venv --python 3.12 # make a python 3.12 virtual environment
source .venv/bin/activate # activate the venv
uv pip install .
cd src/TripoSG # cd into the original repo [its .git dir has been deleted]
python -m scripts.inference_triposg --image-input /full/path/to/your/image.png
```

This will produce a file `output.glb` which can be imported into Blender etc.

### Gradio app

To run just the Gradio app (i.e. the HuggingFace Space):

- Note that the space contains a texture .so
  [here](https://github.com/lmmx/tripos/blob/master/src/space/texture.cpython-310-x86_64-linux-gnu.so)
  which means you should have git LFS installed

```bash
uv venv --python 3.11 # make a python 3.12 virtual environment
source .venv/bin/activate # activate the venv
uv pip install .[space]
cd src/space # cd into the original HF Space repo [its .git dir has been deleted]
python app.py
```

This will run the Gradio GUI app
