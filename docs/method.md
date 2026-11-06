# Méthode

Nous voulions qu'un téléphone ou une Raspberry Pi reconnaisse la maladie d'une feuille sans réseau. Le sujet impose un modèle non trivial sur un appareil embarqué et une à trois optimisations. Nous en avons testé trois : la distillation, la quantification post-entraînement dynamique et la quantification post-entraînement statique avec calibration.

## Données

PlantVillage, dans sa version couleur du dépôt Hugging Face `mohanty/PlantVillage` : 54 305 photos de feuilles, 38 classes (14 cultures, maladies et feuilles saines).

Le piège de ce jeu est connu. Plusieurs photos montrent la même feuille sous des angles voisins. Avec un tirage aléatoire, une feuille se retrouve à la fois en entraînement et en test, et la précision gonfle sans que le modèle ait appris à généraliser. Nous avons donc regroupé les images par feuille, grâce à la table `leaf-map.json` du dépôt, puis découpé avec `StratifiedGroupKFold` (5 plis, graine 42) : un pli pour le test, un pour la validation, trois pour l'entraînement. Un test unitaire vérifie qu'aucune feuille n'apparaît dans deux jeux.

Pour tenir sur un processeur partagé, nous plafonnons chaque classe : 150 images d'entraînement, 40 de validation et 100 de test au plus. Cela donne 5 642, 1 508 et 3 579 images. Aucune feuille n'est commune à deux jeux. Les images sont mises en cache en 224 x 224.

## Prétraitement partagé entre Python et le navigateur

Le navigateur ne reproduit pas le redimensionnement de PIL, et un écart de prétraitement coûte de la précision sans que rien ne plante. Nous avons donc défini le prétraitement comme un algorithme simple, écrit deux fois : recadrage carré centré, réduction par moyenne de surface (chaque pixel de sortie est la moyenne exacte de la zone source qu'il couvre), puis normalisation ImageNet. `src/edgeplant/preprocess.py` et `app/src/preprocess.ts` suivent les mêmes étapes, et `tests/test_web_parity.py` compare leurs sorties sur des images de tailles variées, avec une tolérance de 1e-4.

L'entraînement utilise exactement ce prétraitement, plus des retournements, des quarts de tour, un décalage de 16 pixels et une variation de luminosité et de contraste.

## Modèles

L'élève est un MobileNetV3-Small de torchvision, initialisé sur ImageNet, avec une tête à 38 sorties. Il est ajusté avec AdamW (taux 2e-3, décroissance 1e-4), un calendrier one-cycle, 4 époques, des lots de 32 et un lissage d'étiquettes de 0,1.

Le professeur est un EfficientNet-B0. Ajuster tout le réseau sur un processeur partagé coûtait trop cher : nous gelons les cinq premiers étages, mettons leurs activations en cache une fois, puis ajustons le reste (l'essentiel des paramètres) sur ce cache pendant 6 époques. Le même code tourne sur GPU dans le carnet Colab.

## Distillation

Les logits du professeur sont calculés une fois par image d'entraînement, sans augmentation, puis réutilisés comme cibles douces : perte = 0,7 x T² x KL(professeur, élève) + 0,3 x entropie croisée, avec T = 4. Ce choix a un défaut que nous assumons : l'élève voit une image augmentée alors que la cible douce vient de l'image d'origine. Les retournements et les quarts de tour ne changent pas la classe d'une feuille, l'approximation reste raisonnable.

## Quantification

Elle passe par ONNX Runtime, après `quant_pre_process`. Nous avons testé sept variantes, nommées d'après ce qui reste en flottant :

- `dynamic_gemm` : seul le classifieur est quantifié ;
- `dynamic_all` et `static_all_minmax` : toutes les convolutions et le classifieur, la recette naïve ;
- `dynamic_mixed` et `static_mixed_*` : la convolution d'entrée et les convolutions séparables en profondeur restent en flottant, les 40 convolutions ponctuelles et le classifieur passent en int8.

La version statique utilise le format QDQ, des poids int8 par canal, des activations uint8 et 200 images de calibration tirées de l'entraînement, avec trois méthodes de calibration : min-max, percentile et entropie.

Le choix des îlots flottants vient d'une mesure, pas d'une intuition. La recette naïve effondre la précision. Nous avons alors quantifié une famille de couches à la fois sur les 1 508 images de validation (`python -m edgeplant.ablation`) : seules les convolutions ponctuelles et le classifieur gardent la précision, la convolution d'entrée seule et les convolutions séparables seules la détruisent. Les chiffres complets sont dans `docs/results.md`.

## Mesure de la vitesse

Latence d'une image (taille de lot 1) avec ONNX Runtime sur CPU : 10 passages de chauffe, 100 mesures par tour, 3 tours entrelacés entre les modèles dans un ordre tiré au hasard, médiane des tours. On rapporte p50 et p95. Un thread et plusieurs threads sont mesurés séparément.

La machine de mesure est un poste de travail partagé : d'autres processus tournaient en même temps, les valeurs absolues sont donc bruitées. Les modèles sont mesurés dans les mêmes conditions et entrelacés, ce qui rend leurs rapports plus fiables que les valeurs elles-mêmes.

Pour l'arm64, nous utilisons Docker avec l'émulation QEMU, comme substitut d'une Raspberry Pi. C'est de l'émulation : la latence obtenue n'est pas celle d'une carte réelle et ne sert qu'à comparer les modèles entre eux sur une architecture différente.
