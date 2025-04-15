import torch
import torch.nn as nn
import torch.nn.functional as F

class LossCalculator:
    def __init__(self, pred_y, y, clustering_loss, lambda1=0.00008, lambda2=0.00008):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.y = y
        pred_y = pred_y.reshape(y.shape)
        self.clustering_loss = clustering_loss

    def regression_loss(self, pred_y, y):
        r_loss = nn.MSELoss()
        l_reg = r_loss(pred_y.to(torch.float32), y.to(torch.float32))
        return l_reg

    def temporal_smoothness_loss(self, pred_y):
        length = pred_y.size(0)
        y_i = pred_y[0:length-1]
        y_i_plus_1 = pred_y[1:length]

        squared_diff = torch.pow(y_i - y_i_plus_1, 2)
        sum_squared_diff = torch.sum(squared_diff)
        temp_smooth_loss = (sum_squared_diff)/(length-1)

        return temp_smooth_loss

    def sparsity_loss(self, pred_y):
        return torch.mean(pred_y)

    def total_loss(self, pred_y, y):
        l_reg = self.regression_loss(pred_y, y)
        temp_smooth_loss = self.temporal_smoothness_loss(pred_y)
        spars_loss = self.sparsity_loss(pred_y)
        total_loss = l_reg + self.lambda1*(spars_loss + temp_smooth_loss) + self.lambda2*(self.clustering_loss)

        return total_loss