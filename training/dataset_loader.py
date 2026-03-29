import torch
from torch.utils.data import Dataset
import pandas as pd
from preprocessing.image_preprocess import preprocess_image
from preprocessing.clinical_preprocess import preprocess_clinical
from preprocessing.text_preprocess import preprocess_text

class TBDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image = preprocess_image(row['image_path'])

        clinical_data = {
            "cbc": row['cbc'],
            "esr": row['esr'],
            "crp": row['crp']
        }

        clinical = preprocess_clinical(clinical_data)

        input_ids, attention_mask = preprocess_text(row['history'])

        tb_label = torch.tensor(row['tb_label'])
        active_label = torch.tensor(row['active_label'])
        risk_score = torch.tensor(row['risk_score'], dtype=torch.float32)

        return image, clinical, input_ids.squeeze(0), attention_mask.squeeze(0), tb_label, active_label, risk_score