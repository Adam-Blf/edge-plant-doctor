# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog, versions follow Semantic Versioning.

## [Unreleased]

## [1.0.0] - 2026-11-06

### Added
- Project scaffold, deterministic preprocessing shared by Python and the web app.
- Leaf-grouped, class-stratified PlantVillage split (train 5642, val 1508, test 3579 images, no shared leaf).
- Fine-tuning loop, cached-feature EfficientNet-B0 teacher, distillation loss.
- ONNX export and ONNX Runtime post-training quantization (dynamic, static min-max, percentile, entropy).
- Accuracy evaluation and interleaved latency benchmark, x86 at 1 and 4 threads and arm64 under QEMU (#3).
- Per-family quantization study showing why the stem and depthwise convolutions stay in float (#3).
- Results tables, figures and analysis in `docs/results.md` (#3).
- Colab GPU notebook running the package CLI (#4).
- Offline PWA for iOS and Android with onnxruntime-web, Reicon icons, CI job and GitHub Pages deploy (#5).
- Raspberry Pi inference service and arm64 image, French class labels shared with the apps (#6).
- Native Android app (React Native, Expo, onnxruntime) with the APK built and attached to releases by CI (#7).
- Shipped model: distilled student, static int8 with float stem and depthwise layers, percentile calibration: 98.24 % test accuracy, 2.00 MB.
- MIT licence.
