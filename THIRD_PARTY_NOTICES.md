# Third-party notices

The source code in this repository is released under the [MIT License](LICENSE). That license
covers the code only. The items below belong to their respective owners and keep their own
terms.

## NWPU-RESISC45 dataset

- **Creators:** Gong Cheng, Junwei Han, and Xiaoqiang Lu.
- **Source:** [the dataset authors' page](https://gcheng-nwpu.github.io/), which identifies the
  dataset as a publicly available benchmark and licenses it under
  [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **Alternate copy:** [NWPU-RESISC45 on Figshare](https://figshare.com/articles/dataset/NWPU-RESISC45/19166525?file=34054286)
  (DOI: [10.6084/m9.figshare.19166525.v1](https://doi.org/10.6084/m9.figshare.19166525.v1)). A published
  paper lists this DOI as a location for a copy of NWPU-RESISC45. This is a download reference, not
  the source of the license; the dataset authors' CC BY-NC 4.0 terms above still apply. This project
  does not claim this particular copy was used to produce its results.
- **Citation:** Gong Cheng, Junwei Han, and Xiaoqiang Lu, "Remote Sensing Image Scene
  Classification: Benchmark and State of the Art," *Proceedings of the IEEE*, 105(10),
  1865–1883, 2017. [doi:10.1109/JPROC.2017.2675998](https://doi.org/10.1109/JPROC.2017.2675998).
- **Use limit:** CC BY-NC 4.0 permits sharing and adaptation only for non-commercial purposes,
  subject to attribution, a link to the license, and notice of changes. Commercial use is not
  permitted under this dataset license.

The full dataset is not included. The 8 images in `demo/examples/` are from the test split; they
also appear in `reports/figures/example_predictions.png` and `reports/figures/demo.gif`. The report
figure and GIF resize, crop, arrange, and annotate those images. Those figures are adaptations of
the source images and are covered by the same non-commercial license and attribution terms.

Attribution for these uses: "NWPU-RESISC45, by Gong Cheng, Junwei Han, and Xiaoqiang Lu; used and
adapted under CC BY-NC 4.0." No endorsement by the dataset authors or Northwestern Polytechnical
University is implied.

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
| PyYAML | MIT |
| Gradio (demo only) | Apache-2.0 |
