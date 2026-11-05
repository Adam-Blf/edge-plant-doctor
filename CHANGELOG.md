# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog, versions follow Semantic Versioning.

## [Unreleased]

### Added
- Project scaffold, deterministic preprocessing shared by Python and the web app.
- Leaf-grouped, class-stratified PlantVillage split (train 5642, val 1508, test 3579 images, no shared leaf).
- Fine-tuning loop, cached-feature EfficientNet-B0 teacher, distillation loss.
- ONNX export and ONNX Runtime post-training quantization (dynamic, static min-max, percentile, entropy).
