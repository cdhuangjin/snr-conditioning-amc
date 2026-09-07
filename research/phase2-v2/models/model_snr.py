"""SNR-conditioned AWN: inject a learned SNR embedding before the classifier head.
Based on models/model.py (AWN). Training/eval scripts provide the SNR bin index.
"""
import torch
import torch.nn as nn

from models.lifting import LiftingScheme


class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        global regu_d, regu_c
        (L, H) = self.wavelet(x)
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                regu = regu_d
            elif self.regu_details == 0.0:
                regu = regu_c
            else:
                regu = regu_d + regu_c
            return approx, details, regu


class AWNSNR(nn.Module):
    def __init__(self,
                 num_classes,
                 num_levels=1,
                 in_channels=64,
                 kernel_size=3,
                 latent_dim=320,
                 regu_details=0.01,
                 regu_approx=0.01,
                 num_snr_bins=20,
                 snr_emb_dim=8):
        super(AWNSNR, self).__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = self.in_channels * (self.num_levels + 1)
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.num_snr_bins = num_snr_bins
        self.snr_emb_dim = snr_emb_dim

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.in_channels, kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels, kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            self.levels.add_module('level_' + str(i), LevelTWaveNet(self.in_channels, self.kernel_size,
                                                                    self.regu_details, self.regu_approx))

        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # SNR conditioning: learned embedding of the SNR bin, concatenated with pooled features
        self.snr_embedding = nn.Embedding(num_snr_bins, snr_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels + snr_emb_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, num_classes)
        )

    def forward(self, x, snr_bin=None):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(2)
        x = self.conv2(x)
        regu_sum = []
        det = []
        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)

        if snr_bin is not None:
            emb = self.snr_embedding(snr_bin)
            x = torch.cat([x, emb], 1)

        logit = self.fc(x)
        return logit, regu_sum
