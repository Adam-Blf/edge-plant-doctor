| Modèle | Précision | F1 macro | Taille (Mo) | p50 x86 (ms) | p95 x86 (ms) |
|---|---:|---:|---:|---:|---:|
| Élève seul, FP32 | 97.93 % | 97.91 % | 6.24 | 8.8 | 19.5 |
| Élève seul, Dynamique, tout (naïf) | 2.79 % | 0.14 % | 1.74 | 141.1 | 251.7 |
| Élève seul, Dynamique, classifieur seul | 97.96 % | 97.93 % | 4.37 | 7.6 | 13.2 |
| Élève seul, Dynamique, mixte | 97.93 % | 97.89 % | 1.89 | 62.5 | 88.1 |
| Élève seul, Statique, tout (naïf) | 18.33 % | 16.69 % | 1.89 | 9.8 | 13.2 |
| Élève seul, Statique mixte, entropie | 97.29 % | 97.28 % | 2.00 | 8.7 | 22.4 |
| Élève seul, Statique mixte, min-max | 97.29 % | 97.28 % | 2.00 | 9.0 | 26.6 |
| Élève seul, Statique mixte, percentile | 98.10 % | 98.12 % | 2.00 | 8.1 | 27.3 |
| Élève distillé, FP32 | 98.32 % | 98.33 % | 6.24 | 8.2 | 17.4 |
| Élève distillé, Dynamique, tout (naïf) | 2.91 % | 0.34 % | 1.74 | 119.8 | 277.6 |
| Élève distillé, Dynamique, classifieur seul | 98.30 % | 98.30 % | 4.37 | 8.3 | 15.6 |
| Élève distillé, Dynamique, mixte | 98.02 % | 98.02 % | 1.89 | 60.4 | 88.8 |
| Élève distillé, Statique, tout (naïf) | 35.85 % | 32.27 % | 1.89 | 13.7 | 34.2 |
| Élève distillé, Statique mixte, entropie | 97.63 % | 97.67 % | 2.00 | 8.8 | 22.3 |
| Élève distillé, Statique mixte, min-max | 97.63 % | 97.67 % | 2.00 | 9.0 | 20.8 |
| Élève distillé, Statique mixte, percentile | 98.24 % | 98.22 % | 2.00 | 7.6 | 13.7 |

Latence x86 (AMD64, onnxruntime 1.24.4)

| Modèle | Threads | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Élève seul, Dynamique, tout (naïf) | 1 | 141.1 | 251.7 |
| Élève seul, Dynamique, classifieur seul | 1 | 7.6 | 13.2 |
| Élève seul, Dynamique, mixte | 1 | 62.5 | 88.1 |
| Élève seul, FP32 | 1 | 8.8 | 19.5 |
| Élève seul, Statique, tout (naïf) | 1 | 9.8 | 13.2 |
| Élève seul, Statique mixte, entropie | 1 | 8.7 | 22.4 |
| Élève seul, Statique mixte, min-max | 1 | 9.0 | 26.6 |
| Élève seul, Statique mixte, percentile | 1 | 8.1 | 27.3 |
| Élève distillé, Dynamique, tout (naïf) | 1 | 119.8 | 277.6 |
| Élève distillé, Dynamique, classifieur seul | 1 | 8.3 | 15.6 |
| Élève distillé, Dynamique, mixte | 1 | 60.4 | 88.8 |
| Élève distillé, FP32 | 1 | 8.2 | 17.4 |
| Élève distillé, Statique, tout (naïf) | 1 | 13.7 | 34.2 |
| Élève distillé, Statique mixte, entropie | 1 | 8.8 | 22.3 |
| Élève distillé, Statique mixte, min-max | 1 | 9.0 | 20.8 |
| Élève distillé, Statique mixte, percentile | 1 | 7.6 | 13.7 |
| Élève seul, Dynamique, tout (naïf) | 4 | 96.8 | 231.4 |
| Élève seul, Dynamique, classifieur seul | 4 | 6.7 | 14.4 |
| Élève seul, Dynamique, mixte | 4 | 43.9 | 138.6 |
| Élève seul, FP32 | 4 | 8.9 | 17.8 |
| Élève seul, Statique, tout (naïf) | 4 | 8.3 | 26.9 |
| Élève seul, Statique mixte, entropie | 4 | 7.4 | 18.5 |
| Élève seul, Statique mixte, min-max | 4 | 9.7 | 21.9 |
| Élève seul, Statique mixte, percentile | 4 | 7.0 | 12.7 |
| Élève distillé, Dynamique, tout (naïf) | 4 | 83.5 | 213.8 |
| Élève distillé, Dynamique, classifieur seul | 4 | 6.9 | 11.5 |
| Élève distillé, Dynamique, mixte | 4 | 44.3 | 106.9 |
| Élève distillé, FP32 | 4 | 7.3 | 14.9 |
| Élève distillé, Statique, tout (naïf) | 4 | 9.5 | 19.2 |
| Élève distillé, Statique mixte, entropie | 4 | 10.1 | 28.9 |
| Élève distillé, Statique mixte, min-max | 4 | 8.5 | 26.6 |
| Élève distillé, Statique mixte, percentile | 4 | 6.3 | 11.6 |

Latence arm64 émulée (aarch64, onnxruntime 1.31.1, émulé)

| Modèle | Threads | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Élève seul, FP32 | 1 | 4573.2 | 5144.6 |
| Élève distillé, FP32 | 1 | 3980.6 | 4647.6 |
| Élève distillé, Dynamique, classifieur seul | 1 | 4236.9 | 4773.8 |
| Élève distillé, Statique mixte, percentile | 1 | 2270.2 | 2539.9 |

Une seule famille en int8, le reste en flottant (1508 images de validation, FP32 : 98.21 %)

| Famille quantifiée | Couches | Précision | Accord avec FP32 |
|---|---:|---:|---:|
| Convolution d'entrée | 1 | 25.33 % | 25.60 % |
| Convolutions en profondeur | 11 | 39.79 % | 40.12 % |
| Convolutions ponctuelles | 40 | 98.01 % | 98.67 % |
| Classifieur | 2 | 98.21 % | 100.00 % |
