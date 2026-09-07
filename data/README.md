# Data assets

Large arrays are distributed as assets on release `v0.1.0-research` in this repository. Extract each ZIP into the repository root, preserving paths.

- `radioml2016-data.zip`: unchanged RML2016.10a and RML2016.04C input files. The 10a file is placed in `research/phase2-v2/data/`; 04C in root `data/` as expected by the retained runners. Credit: DeepSig / RadioML, https://www.deepsig.ai/datasets/. These third-party files are **CC BY-NC-SA 4.0**, https://creativecommons.org/licenses/by-nc-sa/4.0/; the repository MIT license does not override that license. Original dataset publications and known errata are linked on the provider page. The files are redistributed without modification; the original filename spelling is retained. No endorsement by DeepSig is implied.
- `independent-waveforms.zip`: the retained six-class mathematical waveform dataset, channel A/B observations, pilot arrays, noise arrays, latent controls, and identities from the manuscript experiment. This release supplies the complete retained arrays, not merely a demonstration subset. Generator: `research/phase2-v2/v2/phase12/core.py`; construction seed 20260906. Project-generated arrays are distributed under the repository MIT license.
- `SHA256SUMS.txt`: hashes of release assets. `data/asset-index.json` records hashes and restored paths of individual raw files.

The published seed summaries are in `results/paper/`. No RML2018.01a training result is claimed. Pretrained neural checkpoints are not included in this release; training code and settings are included. Access to release assets follows the repository visibility and permissions.
