import torch
import timm
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from utils.hparams import hparams
from timm.layers import create_conv2d
from models.sits_branch import SITSSegmenter
from models.fusion_module.aer_cross_sat_atts import FFCA
from models.decoders.unet_former_decoder import UNetFormerDecoder


class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config
        self.embed_dim = config["models"]["t_convformer"]["embed_dim"]
        self.decoder_channels = config["models"]["maxvit"]["decoder_channels"]
        self.num_classes = config["inputs"]["num_classes"]
        self.dropout = config["models"]["maxvit"]["dropout"]
        self.window_size = config["models"]["maxvit"]["window_size"]

        # 1. SITS Encoder
        self.sits_encoder = SITSSegmenter(
            img_size=config["inputs"]["sr_patch_size"],
            in_chans=config["inputs"]["num_channels_sat"],
            embed_dim=config["models"]["t_convformer"]["embed_dim"],
            uper_head_dim=config["models"]["t_convformer"]["uper_head_dim"],
            depths=config["models"]["t_convformer"]["depths"],
            num_heads=config["models"]["t_convformer"]["num_heads"],
            mlp_ratio=config["models"]["t_convformer"]["mlp_ratio"],
            num_classes=config["inputs"]["num_classes"],
            nbts=config["inputs"]["nbts"],
            pool_scales=config["models"]["t_convformer"]["pool_scales"],
            spa_temp_att=config["models"]["t_convformer"]["spa_temp_att"],
            conv_spa_att=config["models"]["t_convformer"]["conv_spa_att"],
            decoder_channels=config["models"]["maxvit"]["decoder_channels"],
            window_size=config["models"]["maxvit"]["window_size"],
            d_model=config["models"]["t_convformer"]["d_model"],
            config=config,
        )

        # # 2. Aerial Network
        self.aerial_net = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=config["inputs"]["num_classes"],
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = (
            self.aerial_net.stem.conv1
        )  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["inputs"][
                "num_channels_aer"
            ],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1,  # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None,
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aerial_net.stem.conv1 = new_conv

        encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        # 3. Decoder from U-Net Former paper
        self.decoder = UNetFormerDecoder(
            encoder_channels,
            self.decoder_channels,
            self.dropout,
            self.window_size,
            self.num_classes,
        )
        self.fusion_module = FFCA(
            aer_channels_list=[128, 256, 512],
            sits_channels_list=[64, 128, 256],
            num_heads=8,
        )

    def forward(
        self,
        img: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        dates: torch.FloatTensor,
    ):

        h, w = img.size()[-2:]
        # Aerial branch
        res0, res1, res2, res3, res4 = self.aerial_net(img)

        # SITS branch
        output_sen, cls_sits, multi_lvls_outs = self.sits_encoder(img_sr, dates)

        # Fusion FFCA
        res2, res3, res4 = self.fusion_module([res2, res3, res4], multi_lvls_outs)

        # Decoder
        logits = self.decoder(res0, res1, res2, res3, res4, h, w)
        return output_sen, cls_sits, logits


# Description of GRID attention introduced in TConvFormer

#  Imagine you have a 6×6 image, and you want each pixel to "see" other pixels globally.

# Step 1: Split into grid

# Divide the 6×6 image into 2×2 grids, so you have 9 grids in total. Each grid has 2×2 pixels.

# Step 2: Grid-attention with dilation

# Instead of computing attention for all 36 pixels (which is expensive), you:
# First compute local attention within each grid (2×2 → small and fast).
# Then compute attention across grids, but using dilated connections (e.g., only attend to every 2nd grid in each direction).
# This way, even distant pixels can influence each other, without doing full 36×36 attention.
