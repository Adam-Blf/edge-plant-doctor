# Plant Doctor sur Raspberry Pi

Ce dossier contient tout ce qu'il faut pour faire tourner le modèle int8 sur une carte arm64 : une image Docker, un service HTTP et une commande pour diagnostiquer une photo depuis le terminal. Le service n'utilise que `onnxruntime`, `numpy` et `pillow`. Il n'embarque ni PyTorch ni code d'entraînement, ce qui garde l'image petite.

Ce paquet n'a pas été lancé sur une vraie Raspberry Pi. Il a été construit et exécuté en `linux/arm64` émulé par QEMU sous Docker, ce qui prouve qu'il s'installe et répond, pas qu'il tient une cadence précise sur la carte. Les latences mesurées sous émulation sont données à titre indicatif dans `docs/results.md`.

## Prérequis

Une Raspberry Pi 3, 4 ou 5 sous un système 64 bits (Raspberry Pi OS 64 bits, Ubuntu Server arm64). Sur un système 32 bits, aucune roue `onnxruntime` récente n'existe, il faut en changer.

## Option A : Docker

L'image est publiée sur GHCR à chaque tag de version.

```bash
# Un jeton protège le service dès qu'il écoute ailleurs qu'en local
export PLANT_DOCTOR_TOKEN="$(openssl rand -hex 24)"

docker run -d --name plant-doctor --restart unless-stopped \
  -e PLANT_DOCTOR_TOKEN \
  -p 127.0.0.1:8080:8080 \
  ghcr.io/adam-blf/edge-plant-doctor-edge:latest
```

Le port est publié sur `127.0.0.1` de la carte. Pour l'ouvrir au réseau local, remplacer par `-p 8080:8080` : le jeton devient alors la seule protection, gardez-le long.

Tester depuis la carte :

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H "Authorization: Bearer $PLANT_DOCTOR_TOKEN" \
  --data-binary @feuille.jpg
```

La réponse donne les trois classes les plus probables, avec le nom français de la plante et de la maladie, et le temps de prétraitement et d'inférence en millisecondes. `GET /health` répond sans jeton, pour la sonde de santé de Docker.

Pour construire l'image soi-même, depuis la racine du dépôt :

```bash
docker buildx build --platform linux/arm64 -f edge/rpi/Dockerfile -t plant-doctor-edge .
```

## Option B : Python directement

```bash
sudo mkdir -p /opt/plant-doctor && sudo chown "$USER" /opt/plant-doctor
git clone https://github.com/Adam-Blf/edge-plant-doctor.git /opt/plant-doctor
cd /opt/plant-doctor
python3 -m venv .venv
.venv/bin/pip install -r edge/rpi/requirements.txt
export PLANT_DOCTOR_HOME=/opt/plant-doctor PYTHONPATH=/opt/plant-doctor/src:/opt/plant-doctor/edge/rpi
```

Diagnostiquer une photo en ligne de commande, avec le temps médian sur dix passages :

```bash
.venv/bin/python -m plantdoctor_edge.cli feuille.jpg --bench 10
```

Lancer le service au démarrage avec systemd :

```bash
echo "PLANT_DOCTOR_TOKEN=$(openssl rand -hex 24)" | sudo tee /etc/plant-doctor.env >/dev/null
sudo chmod 600 /etc/plant-doctor.env
sudo cp edge/rpi/plant-doctor.service /etc/systemd/system/
sudo systemctl enable --now plant-doctor
```

## Ce que le service refuse

Il écoute sur `127.0.0.1` par défaut. Toute autre adresse sans `PLANT_DOCTOR_TOKEN` fait échouer le démarrage. Les corps de requête de plus de 10 Mo reçoivent un 413, les images de plus de 40 millions de pixels et les fichiers illisibles un 400. Une erreur d'inférence renvoie un message générique, la trace reste dans les journaux de la carte.
