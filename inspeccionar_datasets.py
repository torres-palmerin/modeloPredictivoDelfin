# inspeccionar_datasets.py
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def diagnosticar_archivo(nombre_archivo: str):
    path = os.path.join("data", nombre_archivo)
    if not os.path.exists(path):
        logging.error(f"No se encontró el archivo en: {path}")
        return
        
    logging.info(f"=== INSPECCIONANDO: {nombre_archivo} ===")
    try:
        # Leer solo las primeras 3 filas para no saturar memoria
        df = pd.read_excel(path, nrows=3)
        logging.info(f"Columnas detectadas:\n{df.columns.tolist()}\n")
        
        # Verificar tamaño total de registros sin cargar todo si es muy pesado
        df_shape = pd.read_excel(path, usecols=[0])
        logging.info(f"Total de filas estimadas: {len(df_shape)}")
    except Exception as e:
        logging.error(f"Error al leer {nombre_archivo}: {e}")
    print("-" * 60)

if __name__ == "__main__":
    otros_datasets = [
        "07_undergraduate_pathway with degree automaton.xlsx",
        "08_pregrado_posgrado_automata_corregido_validado.xlsx",
        "08_solo_pregrado_automata_corregido_validado_v2.xlsx"
    ]
    
    for dataset in otros_datasets:
        diagnosticar_archivo(dataset)