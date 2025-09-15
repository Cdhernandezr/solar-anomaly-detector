import os
import zipfile
import time
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import seaborn as sns
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

def download_and_extract_dataset():
    """
    Descarga y extrae el dataset de paneles solares infrarrojos
    de una manera robusta y multiplataforma.
    """
    url = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"
    
    # Define las rutas usando la ubicación del script actual (src/) para que siempre funcione
    # os.path.dirname(__file__) obtiene el directorio del script actual (src)
    # '..' sube un nivel al directorio raíz del proyecto
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    zip_path = os.path.join(data_folder, 'solar_dataset.zip')
    # La carpeta que se crea al descomprimir
    final_dataset_path = os.path.join(data_folder, 'InfraredSolarModules')
    
    # 1. Crear la carpeta 'data' si no existe
    os.makedirs(data_folder, exist_ok=True)

    # 2. Verificar si el dataset ya fue extraído para no volver a descargar
    if os.path.exists(final_dataset_path):
        Logger.log("✅ El dataset ya existe. Omitiendo descarga y extracción.", 'success')
        return final_dataset_path

    Logger.log("📥 Descargando dataset...", 'info')
    
    try:
        # 3. Usar requests para la descarga con un stream y barra de progreso (tqdm)
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanza un error si la descarga falla (e.g., 404)
        
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="solar_dataset.zip")
        
        with open(zip_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        Logger.log("📂 Extrayendo archivos...", 'info')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_folder) # Extraer directamente en la carpeta 'data'

        # 4. Opcional: eliminar el archivo .zip después de extraerlo
        os.remove(zip_path)
        
        return final_dataset_path
        
    except requests.exceptions.RequestException as e:
        Logger.log(f"Error de red al descargar el archivo: {e}", 'error')
        raise SystemExit(e) # Termina el script si la descarga falla
    

class SolarDataManager:
    """Gestor de datos para el dataset de paneles solares"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.load_metadata()

    def load_metadata(self):
        """Carga y procesa los metadatos del dataset"""
        metadata_path = os.path.join(self.dataset_path, 'module_metadata.json')
        self.df = pd.read_json(metadata_path, orient='index').sort_index()

        # Construir rutas completas
        self.df['full_path'] = self.df['image_filepath'].apply(
            lambda x: os.path.join(self.dataset_path, x)
        )

        # Crear etiquetas binarias
        self.df['defect_status'] = self.df['anomaly_class'].apply(
            lambda x: 'Defective' if x != 'No-Anomaly' else 'Normal'
        )

        Logger.log(f"Dataset cargado: {len(self.df)} imágenes", 'success')
        self.show_distribution()

    def show_distribution(self):
        """Muestra la distribución de clases en el dataset"""
        Logger.log("📊 Distribución de clases:", 'info')
        distribution = self.df['defect_status'].value_counts()
        for status, count in distribution.items():
            percentage = (count / len(self.df)) * 100
            print(f"  • {status}: {count} ({percentage:.1f}%)")

    def create_stratified_split(self):
        """Crea división estratificada del dataset"""
        # Primera división: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=Config.TEST_RATIO,
            stratify=self.df['defect_status'],
            random_state=Config.RANDOM_SEED
        )

        # Segunda división: train vs val
        val_size_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['defect_status'],
            random_state=Config.RANDOM_SEED
        )

        Logger.log(f"División completada - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", 'success')
        return train_df, val_df, test_df

def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """Split arrays or matrices into random train and test subsets.

    Quick utility that wraps input validation,
    ``next(ShuffleSplit().split(X, y))``, and application to input data
    into a single call for splitting (and optionally subsampling) data into a
    one-liner.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]

    >>> train_test_split(y, shuffle=False)
    [[0, 1, 2], [3, 4]]
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    train, test = ensure_common_namespace_device(arrays[0], train, test)

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )

# Al final del bloque
return train_df, val_df, test_df