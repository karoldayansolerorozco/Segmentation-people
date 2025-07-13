import os
from kaggle.api.kaggle_api_extended import KaggleApi

def ensure_dataset_downloaded():
    dataset_name = "furkankati/person-segmentation-dataset"
    target_dir = "Segmentation-people/data"
    extracted_dir = os.path.join(target_dir, "person-segmentation-dataset")

    # Crear carpeta destino si no existe
    os.makedirs(target_dir, exist_ok=True)

    # Verifica si ya está descargado
    if not os.path.exists(extracted_dir):
        print("Descargando dataset desde Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        print("Descarga completa.")
    else:
        print("El dataset ya existe en Segmentation-people/data.")

# Ejecutar la descarga antes de cargar datos
ensure_dataset_downloaded()


