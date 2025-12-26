import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_pool, max_pool], dim=1)
        out = self.conv(cat_out)
        M = torch.sigmoid(out)
        return x * M


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_channels, reduced, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Conv2d(reduced, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooling = self.avg_global_pooling(x)
        fc1 = self.fc(avg_pooling)
        relu = self.relu(fc1)
        fc2 = self.fc_out(relu)
        sigmoid = self.sigmoid(fc2)
        return x * sigmoid


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        x = self.channel_attention.forward(x)
        x = self.spatial_attention.forward(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x




class ResNet50UNet(nn.Module):
    def __init__(self, n_classes=13, use_cbam=True, dropout_p=0.2, pretrained=True):
        super(ResNet50UNet, self).__init__()
        self.use_cbam = use_cbam
        
        resnet = models.resnet50(pretrained=pretrained)
        
        # ============ ENCODER (ResNet50) ============
        # Stage 1: Initial conv + bn + relu
        self.encoder1 = nn.Sequential(
            resnet.conv1,      # 7x7 conv, 64 channels
            resnet.bn1,
            resnet.relu
        )
        self.cbam_enc1 = CBAM(64, reduction=16) if use_cbam else None
        self.pool1 = resnet.maxpool
        
        # Stage 2-5: ResNet blocks
        self.encoder2 = resnet.layer1  # 256 channels
        self.cbam_enc2 = CBAM(256, reduction=16) if use_cbam else None
        
        self.encoder3 = resnet.layer2  # 512 channels
        self.cbam_enc3 = CBAM(512, reduction=16) if use_cbam else None
        
        self.encoder4 = resnet.layer3  # 1024 channels
        self.cbam_enc4 = CBAM(1024, reduction=16) if use_cbam else None
        
        self.encoder5 = resnet.layer4  # 2048 channels
        self.cbam_enc5 = CBAM(2048, reduction=16) if use_cbam else None
        
        # ============ BOTTLENECK ============
        self.bottleneck = nn.Sequential(
            ConvBlock(2048, 2048, dropout_p=dropout_p * 1.5),
            ConvBlock(2048, 2048, dropout_p=dropout_p * 1.5)
        )
        self.cbam_bottleneck = CBAM(2048, reduction=16) if use_cbam else None
        
        # ============ DECODER ============
        # Decoder 1: 2048 -> 1024
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1_1 = ConvBlock(2048, 1024, dropout_p=dropout_p)  # 2048 = 1024 (up) + 1024 (skip)
        self.dec1_2 = ConvBlock(1024, 1024, dropout_p=dropout_p)
        self.cbam_dec1 = CBAM(1024, reduction=16) if use_cbam else None
        
        # Decoder 2: 1024 -> 512
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec2_1 = ConvBlock(1024, 512, dropout_p=dropout_p)  # 1024 = 512 (up) + 512 (skip)
        self.dec2_2 = ConvBlock(512, 512, dropout_p=dropout_p)
        self.cbam_dec2 = CBAM(512, reduction=16) if use_cbam else None
        
        # Decoder 3: 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_1 = ConvBlock(512, 256, dropout_p=dropout_p)  # 512 = 256 (up) + 256 (skip)
        self.dec3_2 = ConvBlock(256, 256, dropout_p=dropout_p)
        self.cbam_dec3 = CBAM(256, reduction=16) if use_cbam else None
        
        # Decoder 4: 256 -> 64
        self.up4 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec4_1 = ConvBlock(128, 64, dropout_p=dropout_p)  # 128 = 64 (up) + 64 (skip)
        self.dec4_2 = ConvBlock(64, 64, dropout_p=dropout_p)
        self.cbam_dec4 = CBAM(64, reduction=16) if use_cbam else None
        
        # Decoder 5: Final upsampling to original size
        self.up5 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec5 = ConvBlock(64, 64, dropout_p=dropout_p)
        
        # Final output layer
        self.out = nn.Conv2d(64, n_classes, 1)
        
    def center_crop(self, encoder_features, decoder_features):
        """Crop encoder features to match decoder spatial dimensions"""
        _, _, h_enc, w_enc = encoder_features.size()
        _, _, h_dec, w_dec = decoder_features.size()
        
        dh = h_enc - h_dec
        dw = w_enc - w_dec
        
        if dh == 0 and dw == 0:
            return encoder_features
        
        h_start = dh // 2
        w_start = dw // 2
        
        return encoder_features[:, :, h_start:h_start+h_dec, w_start:w_start+w_dec]
    
    def forward(self, x):
        # ============ ENCODER ============
        # Stage 1
        e1 = self.encoder1(x)  # 64 channels
        if self.use_cbam:
            e1 = self.cbam_enc1(e1)
        p1 = self.pool1(e1)
        
        # Stage 2
        e2 = self.encoder2(p1)  # 256 channels
        if self.use_cbam:
            e2 = self.cbam_enc2(e2)
        
        # Stage 3
        e3 = self.encoder3(e2)  # 512 channels
        if self.use_cbam:
            e3 = self.cbam_enc3(e3)
        
        # Stage 4
        e4 = self.encoder4(e3)  # 1024 channels
        if self.use_cbam:
            e4 = self.cbam_enc4(e4)
        
        # Stage 5
        e5 = self.encoder5(e4)  # 2048 channels
        if self.use_cbam:
            e5 = self.cbam_enc5(e5)
        
        # ============ BOTTLENECK ============
        b = self.bottleneck(e5)
        if self.use_cbam:
            b = self.cbam_bottleneck(b)
        
        # ============ DECODER ============
        # Decoder 1: 2048 -> 1024
        d1 = self.up1(b)
        d1 = torch.cat([d1, self.center_crop(e4, d1)], dim=1)
        d1 = self.dec1_1(d1)
        d1 = self.dec1_2(d1)
        if self.use_cbam:
            d1 = self.cbam_dec1(d1)
        
        # Decoder 2: 1024 -> 512
        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.center_crop(e3, d2)], dim=1)
        d2 = self.dec2_1(d2)
        d2 = self.dec2_2(d2)
        if self.use_cbam:
            d2 = self.cbam_dec2(d2)
        
        # Decoder 3: 512 -> 256
        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.center_crop(e2, d3)], dim=1)
        d3 = self.dec3_1(d3)
        d3 = self.dec3_2(d3)
        if self.use_cbam:
            d3 = self.cbam_dec3(d3)
        
        # Decoder 4: 256 -> 64
        d4 = self.up4(d3)
        d4 = torch.cat([d4, self.center_crop(e1, d4)], dim=1)
        d4 = self.dec4_1(d4)
        d4 = self.dec4_2(d4)
        if self.use_cbam:
            d4 = self.cbam_dec4(d4)
        
        # Decoder 5: Final upsampling
        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        
        # Output
        out = self.out(d5)
        return out