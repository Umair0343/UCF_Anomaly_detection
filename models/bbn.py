import torch
import torch.nn as nn
import torch.nn.functional as F

class NSB1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NSB1, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=0)
        return x
    
class NSB2(nn.Module):
    def __init__(self, input_size2, output_size2):
        super(NSB2, self).__init__()
        self.fc = nn.Linear(input_size2, output_size2)

    def forward(self, x2):
        x2 = self.fc(x2)
        x2 = F.softmax(x2, dim=0)
        return x2
    
class BBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.6):
        super(BBN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(output_size, 1)
        self.nsb1 = NSB1(input_size, hidden_size)
        self.nsb2 = NSB2(hidden_size, output_size)

    def forward(self, x):
        x1 = self.fc1(x)
        nsb1 = self.nsb1(x)

        fc1_res = torch.mul(x1, nsb1)
        x3 = F.relu(fc1_res)
        x3 = self.dropout1(x3)

        x4 = self.fc2(x3)
        nsb2 = self.nsb2(x3)

        fc2_res = torch.mul(x4, nsb2)
        x5 = F.relu(fc2_res)
        x5 = self.dropout2(x5)
        x6 = self.fc3(x5)
        x6 = torch.sigmoid(x6)

        return x6