# archive/

This directory contains historical code that was explored during development but is not part of the active pipeline. Files here are kept for reference and to document what was tried.

## Contents

### `RGBD_FusionPredictor_custom_5layer_cnn.py`

An alternative depth branch using a hand-designed 5-layer CNN instead of the custom ResNet-10 used in `phase4_fusion/extension/`. This variant was trained and evaluated but ultimately replaced by the ResNet-10 version, which showed better performance on the LineMod ADD metric.

The file is never imported by any active script. Its weights (if trained) are stored under `weights/fusion_ext/5layer_cnn/`.
