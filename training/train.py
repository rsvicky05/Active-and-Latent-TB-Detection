import torch
from torch.utils.data import DataLoader
from models.tb_model import TBMultimodalModel
from training.dataset_loader import TBDataset
from training.loss_functions import MultiTaskLoss

def train_model(csv_path, epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TBDataset(csv_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TBMultimodalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = MultiTaskLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            image, clinical, input_ids, attention_mask, tb_label, active_label, risk_label = batch

            image = image.to(device)
            clinical = clinical.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tb_label = tb_label.to(device)
            active_label = active_label.to(device)
            risk_label = risk_label.to(device)

            optimizer.zero_grad()

            tb_out, active_out, risk_out, _ = model(image, clinical, input_ids, attention_mask)

            loss = criterion(tb_out, active_out, risk_out,
                             tb_label, active_label, risk_label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "tb_model.pth")