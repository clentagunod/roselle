
from pandas import read_csv
from torch import load
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATASET_DIR = BASE_DIR / "data" / "training_dataset"
ROSELLE_DATASET_DIR = BASE_DIR / "data" / "roselle_dataset"


class DataManager:
    def __init__(self, path: str, dataset_type: str ='tabular'):
        self._path = path
        self._type = dataset_type 

        self._dataset = None 

        self._validate()

    @property 
    def dataset(self):
        return self._dataset

    def _validate(self): 
        path_splitted = self._path.split("/")
        if len(path_splitted) > 1:
            if self._type == 'tabular':
                self._dataset = read_csv(TRAINING_DATASET_DIR / self._path / f'{path_splitted[1]}_training_features.csv')
            elif self._type == 'graph':
                self._dataset = load(TRAINING_DATASET_DIR / self._path / f'{path_splitted[1]}_molecular_graphs.pt', weights_only=False)
            else:
                raise ValueError(f"'{self._type}' is not a valid data type!")
        else:
            if self._type == 'tabular': 
                self._dataset = read_csv(ROSELLE_DATASET_DIR /  f'{self._path}_features.csv')
            elif self._type == 'graph': 
                self._dataset = load(ROSELLE_DATASET_DIR /  f'{self._path}_molecular_graphs.pt', weights_only=False)
            else: 
                raise ValueError(f"'{self._type}' is not a valid data type!")


                
            



