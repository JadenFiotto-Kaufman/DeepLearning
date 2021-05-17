
import numpy as np
import pandas as pd
import torch
from deeplearning.datasets.base import Dataset
from sqlalchemy import create_engine

class FAPersistance(Dataset):


    def __init__(self,
                 paths,
                 columns,
                 target_column,
                 **kwargs):

        super().__init__(**kwargs)

        self.columns = columns
        self.target_column = target_column

        db = create_engine(f'sqlite:///{paths[0]}')
        #self.data = pd.read_sql_query(f"SELECT {', '.join(self.columns + [self.target_column])} FROM top_features INNER JOIN meta_data ON top_features.faFlightID=meta_data.faFlightID", db) 
        self.data = pd.read_sql_query(f"SELECT * FROM meta_data", db) 

        csv_data = pd.read_csv(paths[1])

        self.data = pd.merge(self.data, csv_data, on='faFlightID')

        self.targets = {target : i for i, target in enumerate(self.data[self.target_column].unique())}

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        data = []

        for column in self.columns:

            data.append(torch.from_numpy(np.array(eval(row[column]))).float())


        data = torch.cat(data, dim=0).unsqueeze(dim=0)

        return data, self.targets[row[self.target_column]]


    def __len__(self):
        return len(self.data)

    @staticmethod
    def args(parser):
        parser.add_argument("--paths", nargs='+', type=str, help="path to flight aware db and csv", required=True)

        parser.add_argument("--columns", nargs='+',type=str, required=True ,help="persistance image columns")
        parser.add_argument("--target_column",type=str, required=True ,help="persistance image target column")

        super(FAPersistance,
              FAPersistance).args(parser)
