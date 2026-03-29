import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, tb_out, active_out, risk_out,
                tb_label, active_label, risk_label):

        loss_tb = self.ce(tb_out, tb_label)
        loss_active = self.ce(active_out, active_label)
        loss_risk = self.mse(risk_out.squeeze(), risk_label)

        total_loss = loss_tb + loss_active + loss_risk

        return total_loss