import torch
import torch.nn as nn
import torch.nn.functional as F

class TacticalZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(7, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = self._make_res_block(64)
        self.res2 = self._make_res_block(64)
        
        # Policy Head
        self.p_conv = nn.Conv2d(64, 32, 1)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 3 * 3, 18)
        
        # Value Head
        self.v_conv = nn.Conv2d(64, 16, 1)
        self.v_bn = nn.BatchNorm2d(16)
        self.v_fc1 = nn.Linear(16 * 3 * 3, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(x + self.res1(x))
        x = F.relu(x + self.res2(x))
        
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(-1, 32 * 3 * 3)
        p = F.log_softmax(self.p_fc(p), dim=1)
        
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(-1, 16 * 3 * 3)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        
        return p, v