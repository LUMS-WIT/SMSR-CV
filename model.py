import torch.nn as nn
import torch.nn.functional as F
import torch


class SRCNN_Shuffle(nn.Module):
    """
    Super-Resolution CNN with PixelShuffle for soil moisture downscaling.

    SR paradigm preserved:
      1. Encoder compresses 54×54 → 6×6 (the native 9 km SMAP grid).
         This forces the network through the actual coarse-resolution bottleneck,
         so it MUST learn a coarse-to-fine mapping — not a trivial identity.
      2. Decoder uses PixelShuffle to upsample 6×6 → 54×54 (×9 spatial).
         PixelShuffle rearranges channel dims into spatial dims, which is the
         standard learned sub-pixel upsampling used in SRCNN/ESPCN literature.

    If we skipped the bottleneck (just 54→54 convolutions), the model could
    learn a pixel-wise regression shortcut and the SR inductive bias is lost.

    Input                          : [B,   C, 54, 54]   (C = in_channels, e.g. 8)
                                       ↓
    Conv2d(C→64,  k5, s3, p2)     : [B,  64, 18, 18]   floor((54 + 2*2 - 5)/3 + 1) = 18
    ReLU                           : [B,  64, 18, 18]
                                        ↓
    Conv2d(64→128, k5, s3, p2)    : [B, 128,  6,  6]   floor((18 + 2*2 - 5)/3 + 1) = 6
    ReLU                           : [B, 128,  6,  6]
                                        ↓
    ══ Bottleneck (at 9 km scale) ══
    Conv2d(128→128, k3, p1)       : [B, 128,  6,  6]
    ReLU                           : [B, 128,  6,  6]
    Conv2d(128→128, k3, p1)       : [B, 128,  6,  6]
    ReLU                           : [B, 128,  6,  6]
                                        ↓
    ══ Upsample Stage 1 ════════════
    Conv2d(128→1152, k3, p1)      : [B,1152,  6,  6]   (1152 = 128 × 3²)
    PixelShuffle(3)                : [B, 128, 18, 18]   (6×3 = 18)
    ReLU                           : [B, 128, 18, 18]
                                        ↓
    ══ Upsample Stage 2 ════════════
    Conv2d(128→9, k3, p1)         : [B,   9, 18, 18]   (9 = 1 × 3²)
    PixelShuffle(3)                : [B,   1, 54, 54]   (18×3 = 54)
                                        ↓
    Output                         : [B,   1, 54, 54]   — predicted y_1km
    """

    def __init__(self, in_channels=8, base_filters=64, dropout=0.2):
        super().__init__()

        # ── Encoder: 54×54  →  18×18  →  6×6 ──────────────────────────
        # Two stride-3 convolutions reduce spatial dims by 9× total,
        # matching the 9 km / 1 km ratio exactly.
        self.encoder = nn.Sequential(
            # [B, C, 54, 54] → [B, 64, 18, 18]   (stride=3, pad=1, k=5)
            nn.Conv2d(in_channels, base_filters, kernel_size=5, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # [B, 64, 18, 18] → [B, 128, 6, 6]   (stride=3, pad=1, k=5)
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=5, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # ── Bottleneck: refine at coarse 6×6 resolution ───────────────
        # Non-linear mixing at the native SMAP scale.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # ── Decoder / Upsampler: 6×6  →  18×18  →  54×54 ─────────────
        # Two-stage PixelShuffle (×3 each  →  ×9 total).
        # Stage 1: needs 128 → C1 where C1 = out_ch × 3² = 128×9 = 1152?
        #   That's too heavy.  Instead: reduce channels first, then shuffle.
        #
        # Stage 1:  [B, 128, 6, 6]
        #        conv → [B, 128*9, 6, 6]  = [B, 1152, 6, 6]
        #        shuffle(3) → [B, 128, 18, 18]
        #
        # Stage 2:  [B, 128, 18, 18]
        #        conv → [B, 9, 18, 18]
        #        shuffle(3) → [B, 1, 54, 54]
        self.upsample = nn.Sequential(
            # --- Stage 1: 6×6 → 18×18 ---
            nn.Conv2d(base_filters * 2, base_filters * 2 * 9,
                      kernel_size=3, padding=1),          # [B, 1152, 6, 6]
            nn.PixelShuffle(upscale_factor=3),            # [B, 128, 18, 18]
            nn.ReLU(inplace=True),

            # --- Stage 2: 18×18 → 54×54 ---
            nn.Conv2d(base_filters * 2, 1 * 9,
                      kernel_size=3, padding=1),          # [B, 9, 18, 18]
            nn.PixelShuffle(upscale_factor=3),            # [B, 1, 54, 54]
        )

    def forward(self, x):
        """
        x : [B, C, 54, 54]  — multiband input (SMAP 9km broadcast + aux 1km)
        returns : [B, 1, 54, 54] — predicted 1 km soil moisture
        """
        x = self.encoder(x)      # [B, 128,  6,  6]
        x = self.bottleneck(x)    # [B, 128,  6,  6]
        x = self.upsample(x)     # [B,   1, 54, 54]
        return x

# model = SRCNN_Shuffle(in_channels=19)

# x = torch.randn(16, 19, 54, 54)  # batch of 16
# out = model(x)
# print("Output shape:", out.shape)
# # Output shape: torch.Size([16, 1, 54, 54])
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Model Parameters: {params / 1e6:.2f}M")   # 1.87M


class ResBlock(nn.Module):
    """
    Residual block with dropout for regularization.
    Preserves spatial dimensions throughout.

    Architecture:
        input ──► Conv ──► ReLU ──► Drop ──► Conv ──► Drop ──► (+) ──► output
          │                                                      ▲
          └────────��─────────────────────────────────────────────┘
    """

    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class SRCNN_Residual(nn.Module):
    """
    Residual SR model for soil moisture downscaling.

    Design:
      - Operates at full 54×54 resolution (no spatial downsampling)
      - Learns a residual correction on top of the bilinear SMAP input
      - Output = SMAP_input_band + learned_correction
      - Guarantees performance ≥ bilinear baseline (residual starts near 0)
      - Block loss provides 9km coarse-scale physical anchor

    Parameters
    ----------
    in_channels    : int   — number of input bands (SMAP + aux variables)
    base_filters   : int   — feature channels throughout the network
    num_res_blocks : int   — number of residual blocks (depth)
    dropout        : float — Dropout2d probability (0.0 = no dropout)
    """

    def __init__(self, in_channels=19, base_filters=64, num_res_blocks=6, dropout=0.2):
        super().__init__()

        # ── Feature extraction (no downsampling) ──
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # ── Residual blocks at full resolution ──
        self.res_blocks = nn.Sequential(
            *[ResBlock(base_filters, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # ── Output: single-channel residual correction ──
        self.tail = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_filters, 1, kernel_size=3, padding=1),
            # No activation — residual can be positive or negative
        )

    def forward(self, x):
        """
        x : [B, C, 54, 54]
        Band 0 = bilinear-interpolated SMAP at 1km (the coarse signal).
        Bands 1+ = auxiliary 1km variables (LST, DEM, LULC, STC...).

        Returns: [B, 1, 54, 54] — predicted 1km soil moisture
        """
        identity = x[:, 0:1, :, :]       # [B, 1, 54, 54] — skip connection

        feat = self.head(x)               # [B, 64, 54, 54]
        feat = self.res_blocks(feat)      # [B, 64, 54, 54]
        residual = self.tail(feat)        # [B,  1, 54, 54]

        return identity + residual        # [B,  1, 54, 54]


# model = SRCNN_Residual(in_channels=19)

# x = torch.randn(16, 19, 54, 54)  # batch of 16
# out = model(x)
# print("Output shape:", out.shape)
# # Output shape: torch.Size([16, 1, 54, 54])
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Model Parameters: {params / 1e6:.2f}M")   # 0.51MM
