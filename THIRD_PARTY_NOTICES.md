# Third-party notices

The source code in this repository is released under the [MIT License](LICENSE). That license
covers the code only. The items below belong to their respective owners and keep their own
terms.

## NWPU-RESISC45 dataset

- **Creators:** Gong Cheng, Junwei Han and Xiaoqiang Lu, Northwestern Polytechnical University
- **Paper:** "Remote Sensing Image Scene Classification: Benchmark and State of the Art",
  *Proceedings of the IEEE*, 105(10), 1865–1883, 2017.
  [doi:10.1109/JPROC.2017.2675998](https://doi.org/10.1109/JPROC.2017.2675998)
- **License:** not published by the authors. Mirrors list it as "unknown".

The dataset is **not** included in or distributed with this repository. You have to download it
from the authors' source yourself.

A small number of dataset images (8 test-split images in `demo/examples/`, also shown in
`reports/figures/example_predictions.png` and `reports/figures/demo.gif`) are included to
illustrate the model's output. They remain the property of their rights holders, are shown for
non-commercial research and educational illustration only, and are **not** covered by this
repository's MIT License.

If you hold rights to any of these images and want them removed, please open an issue or use the
contact in [SECURITY.md](SECURITY.md), and they will be removed promptly.

## Pretrained weights

The model is initialised from Keras' ImageNet-pretrained ResNet-50 weights, which are ported from
the weights released by Kaiming He under the MIT License
([source](https://github.com/fchollet/deep-learning-models#licensing)). The ImageNet dataset has
its own [terms of access](https://image-net.org/download).

The trained model produced by this project is **not** included in this repository.

## Software dependencies

Installed separately via `pip`; none are bundled here.

| Package | License |
|---|---|
| TensorFlow, Keras | Apache-2.0 |
| NumPy | BSD-3-Clause (with bundled MIT, Zlib, 0BSD and CC0 components) |
| pandas, scikit-learn | BSD-3-Clause |
| Matplotlib | Matplotlib license (PSF-based) |
| Pillow | MIT-CMU |
| PyYAML, FastAPI, Pydantic | MIT |
| Uvicorn, Starlette | BSD-3-Clause |
| Gradio, python-multipart | Apache-2.0 |

The Docker image is based on the official `python:3.11-slim` image (Python Software Foundation
License, plus Debian package licenses).
