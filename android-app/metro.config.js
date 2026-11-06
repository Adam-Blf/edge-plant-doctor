const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// The ONNX model is bundled as an asset and resolved with expo-asset.
config.resolver.assetExts.push("onnx");

module.exports = config;
