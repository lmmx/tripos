# TripoSG

The [TripoSG](https://github.com/VAST-AI-Research/TripoSG/) project,
packaged with uv for easier installation.

To run:

```bash
uv venv # make a python 3.12 virtual environment
source .venv/bin/activate # activate the venv
uv sync # or uv pip install .
cd TripoSG # cd into the original repo [its .git dir has been deleted]
python -m scripts.inference_triposg --image-input /full/path/to/your/image.png
```

This will produce a file `output.glb` which can be imported into Blender etc.
