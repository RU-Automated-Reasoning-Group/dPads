import torch
import torch.nn as nn
import pdb

class SoftF1LossWithLogits(nn.Module):
    def __init__(self, weight=[1.0, 1.0], double=True):
        super(SoftF1LossWithLogits, self).__init__()
        self.double = double
        self.p_n_w = weight
        if self.p_n_w is not None:
            assert len(self.p_n_w) == 2

        self.sigmoid = nn.Sigmoid()

    def __call__(self, y_hat, y):
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        y_hat = self.sigmoid(y_hat)

        if self.double:
            return self.macro_double_soft_f1(y_hat, y)
        else:
            return self.macro_soft_f1(y_hat, y)


    # soft f1 loss
    def macro_soft_f1(self, y_hat, y):
        """Compute the macro soft F1-score as a cost.
        Average (1 - soft-F1) across all labels.
        Use probability values instead of binary predictions.
        
        Args:
            y (float32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        tp = torch.sum(y * y_hat, dim=0)
        fp = torch.sum((1-y) * y_hat, dim=0)
        fn = torch.sum(y * (1-y_hat), dim=0)
        soft_f1 = 2*tp / (2*tp +fn + fp + 1e-16)

        soft_f1_loss = 1 - soft_f1
        soft_f1_loss = torch.mean(soft_f1_loss)

        return soft_f1_loss


    # soft f1 loss consider class 1 and class 0
    def macro_double_soft_f1(self, y_hat, y):
        """
        Args:
            y (float32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
            
        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        tp = torch.sum(y * y_hat, dim=0)
        fp = torch.sum((1-y) * y_hat, dim=0)
        fn = torch.sum(y * (1-y_hat), dim=0)
        tn = torch.sum((1-y) * (1-y_hat), dim=0)
        soft_f1_p = 2*tp / (2*tp + fn + fp + 1e-16)
        soft_f1_n = 2*tn / (2*tn + fn + fp + 1e-16)

        soft_f1_loss = self.p_n_w[1] * (1 - soft_f1_p) + self.p_n_w[0] * (1 - soft_f1_n)
        soft_f1_loss = torch.mean(soft_f1_loss)

        return soft_f1_loss
