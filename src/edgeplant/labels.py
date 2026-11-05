"""The 38 PlantVillage classes with French display names, shipped to the web app."""

import json
from pathlib import Path

CROPS_FR = {
    "Apple": "Pomme",
    "Blueberry": "Myrtille",
    "Cherry_(including_sour)": "Cerise",
    "Corn_(maize)": "Maïs",
    "Grape": "Vigne",
    "Orange": "Oranger",
    "Peach": "Pêcher",
    "Pepper,_bell": "Poivron",
    "Potato": "Pomme de terre",
    "Raspberry": "Framboisier",
    "Soybean": "Soja",
    "Squash": "Courge",
    "Strawberry": "Fraisier",
    "Tomato": "Tomate",
}

CONDITIONS_FR = {
    "Apple_scab": "Tavelure",
    "Black_rot": "Pourriture noire",
    "Cedar_apple_rust": "Rouille grillagée",
    "healthy": "Feuille saine",
    "Powdery_mildew": "Oïdium",
    "Cercospora_leaf_spot Gray_leaf_spot": "Cercosporiose (tache grise)",
    "Common_rust_": "Rouille commune",
    "Northern_Leaf_Blight": "Helminthosporiose du Nord",
    "Esca_(Black_Measles)": "Esca (rougeot)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Brûlure des feuilles (Isariopsis)",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (jaunissement des agrumes)",
    "Bacterial_spot": "Tache bactérienne",
    "Early_blight": "Alternariose",
    "Late_blight": "Mildiou",
    "Leaf_scorch": "Brûlure des feuilles",
    "Leaf_Mold": "Cladosporiose",
    "Septoria_leaf_spot": "Septoriose",
    "Spider_mites Two-spotted_spider_mite": "Acariens (tétranyque à deux points)",
    "Target_Spot": "Tache cible",
    "Tomato_Yellow_Leaf_Curl_Virus": "Virus de l'enroulement jaune (TYLCV)",
    "Tomato_mosaic_virus": "Virus de la mosaïque",
}

CLASSES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)


def label_entries() -> list[dict]:
    """One record per class index, in model output order."""
    entries = []
    for name in CLASSES:
        crop, condition = name.split("___")
        entries.append(
            {
                "id": name,
                "crop": CROPS_FR[crop],
                "condition": CONDITIONS_FR[condition],
                "healthy": condition == "healthy",
            }
        )
    return entries


def write_labels(path: Path) -> None:
    path.write_text(json.dumps(label_entries(), ensure_ascii=False, indent=1), encoding="utf-8")
