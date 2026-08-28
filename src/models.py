"""Arquitecturas copiadas del notebook principal sin cambios funcionales."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SwinUNet(nn.Module):
    def __init__(self, in_ch=4, encoder_name="swin_tiny_patch4_window7_224", pretrained=True, img_size=128):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch,
            img_size=img_size,
        )
        enc_chs = self.encoder.feature_info.channels()

        self.bottleneck = nn.Sequential(
            ConvBNAct(enc_chs[-1], 256),
            ConvBNAct(256, 256),
        )

        self.dec3 = DecoderBlock(256, enc_chs[-2], 256)
        self.dec2 = DecoderBlock(256, enc_chs[-3], 128)
        self.dec1 = DecoderBlock(128, enc_chs[-4], 64)

        self.head = nn.Sequential(
            ConvBNAct(64, 32),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x_in_h, x_in_w = x.shape[-2], x.shape[-1]

        feats = self.encoder(x)
        feats = [f.permute(0, 3, 1, 2) for f in feats]
        f1, f2, f3, f4 = feats

        x = self.bottleneck(f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        x = self.head(x)
        x = F.interpolate(x, size=(x_in_h, x_in_w), mode="bilinear", align_corners=False)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """
    Residual attention gate.
    g = gating del decoder; x = skip del encoder.
    Usamos x * (1 + psi) para no borrar por completo detalles utiles del skip.
    """
    def __init__(self, F_g, F_l, F_int, residual=True):
        super().__init__()
        self.residual = residual
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        if self.residual:
            return x * (1.0 + psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, up_mode="bilinear", residual_attention=True):
        super().__init__()
        self.up_mode = up_mode

        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch*8, base_ch*16)

        self.up4 = self._make_up(base_ch*16, base_ch*8)
        self.att4 = AttentionGate(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4, residual=residual_attention)
        self.dec4 = DoubleConv(base_ch*16, base_ch*8)

        self.up3 = self._make_up(base_ch*8, base_ch*4)
        self.att3 = AttentionGate(F_g=base_ch*4, F_l=base_ch*4, F_int=base_ch*2, residual=residual_attention)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4)

        self.up2 = self._make_up(base_ch*4, base_ch*2)
        self.att2 = AttentionGate(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch, residual=residual_attention)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)

        self.up1 = self._make_up(base_ch*2, base_ch)
        self.att1 = AttentionGate(F_g=base_ch, F_l=base_ch, F_int=max(1, base_ch//2), residual=residual_attention)
        self.dec1 = DoubleConv(base_ch*2, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def _make_up(self, in_ch, out_ch):
        if self.up_mode == "transpose":
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.out(d1)


class HRFuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HRNetRegressor(nn.Module):
    def __init__(self, in_ch=4, backbone="hrnet_w18_small", pretrained=True, fusion_ch=64):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_ch,
        )
        self.feature_channels = self.backbone.feature_info.channels()
        self.proj = nn.ModuleList([
            HRFuseBlock(ch, fusion_ch) for ch in self.feature_channels
        ])
        self.head = nn.Sequential(
            nn.Conv2d(fusion_ch * len(self.feature_channels), 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _to_nchw(self, feat, expected_ch):
        if feat.ndim != 4:
            raise ValueError(f"HRNet feature ndim inesperado: {feat.ndim}")
        if feat.shape[1] == expected_ch:
            return feat
        if feat.shape[-1] == expected_ch:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"No puedo inferir canales HRNet: shape={tuple(feat.shape)}, expected_ch={expected_ch}")

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        feats = self.backbone(x)

        fused = []
        for feat, expected_ch, proj in zip(feats, self.feature_channels, self.proj):
            feat = self._to_nchw(feat, expected_ch)
            feat = proj(feat)
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            fused.append(feat)

        y = self.head(torch.cat(fused, dim=1))
        return y


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, bilinear=True, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*16)

        self.up1 = Up(base_ch*16, base_ch*8, base_ch*8, bilinear=bilinear)
        self.up2 = Up(base_ch*8, base_ch*4, base_ch*4, bilinear=bilinear)
        self.up3 = Up(base_ch*4, base_ch*2, base_ch*2, bilinear=bilinear)
        self.up4 = Up(base_ch*2, base_ch, base_ch, bilinear=bilinear)

        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (nested skip connections) para regresión DSM.
    Entrada: (B, in_ch, H, W)
    Salida:  (B, 1, H, W)
    """
    def __init__(self, in_ch=4, base_ch=64, out_ch=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.conv00 = ConvBlock(in_ch, base_ch)
        self.conv10 = ConvBlock(base_ch, base_ch*2)
        self.conv20 = ConvBlock(base_ch*2, base_ch*4)
        self.conv30 = ConvBlock(base_ch*4, base_ch*8)
        self.conv40 = ConvBlock(base_ch*8, base_ch*16)

        self.conv01 = ConvBlock(base_ch + base_ch*2, base_ch)
        self.conv11 = ConvBlock(base_ch*2 + base_ch*4, base_ch*2)
        self.conv21 = ConvBlock(base_ch*4 + base_ch*8, base_ch*4)
        self.conv31 = ConvBlock(base_ch*8 + base_ch*16, base_ch*8)

        self.conv02 = ConvBlock(base_ch + base_ch + base_ch*2, base_ch)
        self.conv12 = ConvBlock(base_ch*2 + base_ch*2 + base_ch*4, base_ch*2)
        self.conv22 = ConvBlock(base_ch*4 + base_ch*4 + base_ch*8, base_ch*4)

        self.conv03 = ConvBlock(base_ch + base_ch + base_ch + base_ch*2, base_ch)
        self.conv13 = ConvBlock(base_ch*2 + base_ch*2 + base_ch*2 + base_ch*4, base_ch*2)

        self.conv04 = ConvBlock(base_ch + base_ch + base_ch + base_ch + base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        x01 = self.conv01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x11 = self.conv11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x21 = self.conv21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x31 = self.conv31(torch.cat([x30, self._up(x40, x30)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self._up(x31, x20)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self._up(x22, x10)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self._up(x13, x00)], dim=1))
        return self.outc(x04)


def _constructor(model_class: type[nn.Module], **fixed_kwargs) -> Callable[..., nn.Module]:
    def build(*, pretrained: bool = False) -> nn.Module:
        kwargs = dict(fixed_kwargs)
        if model_class in {SwinUNet, HRNetRegressor}:
            kwargs["pretrained"] = pretrained
        return model_class(**kwargs)

    return build


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "U-Net": {
        "name": "U-Net",
        "constructor": _constructor(UNet, in_ch=4, base_ch=64),
        "short_name": "unet",
        "uses_pretrained": False,
    },
    "U-Net++": {
        "name": "U-Net++",
        "constructor": _constructor(UNetPlusPlus, in_ch=4, base_ch=64),
        "short_name": "unetpp",
        "uses_pretrained": False,
    },
    "Attention-U-Net-Residual": {
        "name": "Attention-U-Net-Residual",
        "constructor": _constructor(AttentionUNet, in_ch=4, base_ch=64),
        "short_name": "attention_unet_residual",
        "uses_pretrained": False,
    },
    "Swin-Tiny-Encoder-CNN-Decoder": {
        "name": "Swin-Tiny-Encoder-CNN-Decoder",
        "constructor": _constructor(SwinUNet, in_ch=4, img_size=128),
        "short_name": "swin_unet",
        "uses_pretrained": True,
    },
    "HRNet-W18-Multiscale": {
        "name": "HRNet-W18-Multiscale",
        "constructor": _constructor(HRNetRegressor, in_ch=4, backbone="hrnet_w18_small"),
        "short_name": "hrnet_w18_multiscale",
        "uses_pretrained": True,
    },
}


def create_model(model_name: str, *, use_pretrained: bool = False) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido {model_name!r}. Disponibles: {list(MODEL_REGISTRY)}"
        )
    spec = MODEL_REGISTRY[model_name]
    pretrained = bool(use_pretrained and spec["uses_pretrained"])
    return spec["constructor"](pretrained=pretrained)


@torch.no_grad()
def sanity_check_models(device=None) -> list[dict[str, Any]]:
    """Comprueba la salida nativa sin interpolación ni corrección posterior."""
    device = torch.device("cpu") if device is None else torch.device(device)
    expected_input = (2, 4, 128, 128)
    expected_output = (2, 1, 128, 128)
    report: list[dict[str, Any]] = []

    for model_name in MODEL_REGISTRY:
        model = None
        output = None
        try:
            model = create_model(model_name, use_pretrained=False).to(device)
            model.eval()
            dummy = torch.randn(*expected_input, device=device)
            output = model(dummy)
            native_shape = tuple(output.shape)
            if native_shape != expected_output:
                raise AssertionError(
                    f"Salida nativa {native_shape}; se esperaba {expected_output}"
                )
            report.append({
                "model": model_name,
                "input_shape": expected_input,
                "native_output_shape": native_shape,
                "ok": True,
                "error": None,
            })
        except Exception as exc:
            report.append({
                "model": model_name,
                "input_shape": expected_input,
                "native_output_shape": None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            if output is not None:
                del output
            if model is not None:
                model.to("cpu")
                del model
            if "dummy" in locals():
                del dummy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return report
