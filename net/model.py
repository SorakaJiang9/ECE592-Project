import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net.blocks import * 
dtype = torch.cuda.FloatTensor


# ------------------------------
# Basic BN + ReLU + Conv block
# Used by LLE, AFFB, and DRH
# ------------------------------
class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, 3, 1, 1))


# ------------------------------
# BN + ReLU + PSSM branch
# This wraps a Pyramidal Selective Scan Module (PSSM)
# ------------------------------
class BNReLUPSSM(nn.Sequential):
    def __init__(self, in_channels, channels, drop_rate, H, W, inplace=True):
        super(BNReLUPSSM, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        # SF_Block is used here as the implementation of the PSSM branch
        self.add_module('pssm', SF_Block(in_channels, channels, drop_rate, H, W))


# ------------------------------
# Gating unit: BN + ReLU + 1x1 Conv
# Used to fuse short-term and long-term features
# ------------------------------
class GateUnit(nn.Sequential):
    def __init__(self, in_channels, channels, inplace=True):
        super(GateUnit, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, 1, 1, 0))


# ------------------------------
# Dual-Domain Fusion Block (DDFB)
# Here it is implemented as two stacked PSSM branches with a residual connection
# ------------------------------
class DDFB(nn.Module):
    def __init__(self, channels, drop_rate, H, W):
        super(DDFB, self).__init__()
        self.pssm_1 = BNReLUPSSM(channels, channels, drop_rate, H, W, True)
        self.pssm_2 = BNReLUPSSM(channels, channels, drop_rate, H, W, True)

    def forward(self, x):
        residual = x
        out = self.pssm_1(x)
        out = self.pssm_2(out)
        out = out + residual
        return out


# ------------------------------
# Multi-Stage Spatial–Spectral Block (MSSB)
# Stacks several DDFBs and fuses their outputs with long-term memory
# using a gating unit
# ------------------------------
class MSSB(nn.Module):
    """
    channels: feature channel dimension
    num_ddfb: number of DDFBs inside this MSSB
    num_memblock: index of this MSSB in the dense chain (1-based)
    H, W: spatial size of the features at this level
    """
    def __init__(self, channels, num_ddfb, num_memblock, drop_rate, H, W):
        super(MSSB, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [DDFB(channels, drop_rate, H, W) for _ in range(num_ddfb)]
        )
        # GateUnit input channels: short-term + long-term features
        self.gate_unit = GateUnit((num_ddfb + num_memblock) * channels, channels, True)

    def forward(self, x, ys):
        """
        x: input feature map for this MSSB
        ys: list of long-term features from previous MSSBs
        Returns the gated output, and appends it to ys as new long-term memory.
        """
        xs = []
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        # Concatenate short-term (xs) and long-term (ys) features along channel dimension
        gate_out = self.gate_unit(torch.cat(xs + ys, dim=1))
        ys.append(gate_out)
        return gate_out


# ============================================================
# Dual-Domain Underwater Enhancement Network (DD-UIE)
#
# Components:
#  - LLE: Low-Level Encoder (lle_conv1/2/3 + pooling)
#  - MSSB chain: deep feature refinement with dense connections
#  - DFAM: Deep Feature Aggregation Module (weighted sum of MSSB outputs)
#  - AFFB: Adaptive Feature Fusion Block (affb_fusion)
#  - DRH: Decoder Reconstruction Head (drh_conv1/2/3)
# ============================================================
class DD_UIE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=16,
        num_memblock=6,   # number of MSSBs
        num_resblock=6,   # number of DDFBs inside each MSSB
        drop_rate=0.0,
        H=256,
        W=256
    ):
        super(DD_UIE, self).__init__()

        # ---------------- Low-Level Encoder (LLE) ----------------
        self.lle_conv1 = BNReLUConv(in_channels, channels, True)
        self.lle_conv2 = BNReLUConv(channels, channels * 2, True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lle_conv3 = BNReLUConv(channels * 2, channels * 4, True)

        # ---------------- Decoder Reconstruction Head (DRH) ----------------
        self.drh_conv1 = BNReLUConv(channels * 4, channels * 2, True)
        self.drh_conv2 = BNReLUConv(channels * 2, channels, True)
        self.drh_conv3 = BNReLUConv(channels, in_channels, True)

        # ---------------- Adaptive Feature Fusion Block (AFFB) ----------------
        # This convolution refines the features at the deepest level
        self.affb_fusion = BNReLUConv(channels * 4, channels * 4, True)

        # ---------------- Deep Feature Aggregation Module (DFAM) ----------------
        # A chain of MSSBs operating on the deepest feature level
        self.mssb_blocks = nn.ModuleList(
            [
                MSSB(channels * 4, num_resblock, i + 1, drop_rate, H // 4, W // 4)
                for i in range(num_memblock)
            ]
        )

        # Learnable weights to aggregate outputs from different MSSBs
        self.dfam_weights = nn.Parameter(
            (torch.ones(1, num_memblock) / num_memblock), requires_grad=True
        )

    def forward(self, x):
        # ---------- LLE: encode low-level and mid-level features ----------
        residual0 = x

        out = self.lle_conv1(x)
        residual1 = out

        out = self.lle_conv2(out)
        out = self.pool(out)
        residual2 = out

        out = self.lle_conv3(out)
        out = self.pool(out)
        residual3 = out

        # ---------- DFAM + MSSB chain: deep feature aggregation ----------
        w_sum = self.dfam_weights.sum(1)
        mid_feat = []  # outputs from each MSSB
        ys = [out]     # long-term memory list, initialized with the deepest LLE feature

        for mssb in self.mssb_blocks:
            out = mssb(out, ys)
            mid_feat.append(out)

        # ---------- AFFB: weighted fusion of MSSB outputs ----------
        # Each MSSB output is fused with the deepest residual feature (residual3)
        # and aggregated using dfam_weights
        pred = (self.affb_fusion(mid_feat[0]) + residual3) * \
               self.dfam_weights.data[0][0] / w_sum

        for i in range(1, len(mid_feat)):
            pred = pred + (self.affb_fusion(mid_feat[i]) + residual3) * \
                   self.dfam_weights.data[0][i] / w_sum

        # ---------- DRH: decode back to image resolution ----------
        pred = F.interpolate(pred, scale_factor=2)
        pred = self.drh_conv1(pred)
        pred = residual2 + pred

        pred = F.interpolate(pred, scale_factor=2)
        pred = self.drh_conv2(pred)
        pred = residual1 + pred

        pred = self.drh_conv3(pred)
        pred = residual0 + pred

        return pred
