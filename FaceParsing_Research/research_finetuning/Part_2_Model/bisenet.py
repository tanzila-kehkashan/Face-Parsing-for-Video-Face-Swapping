"""
BiSeNet: Bilateral Segmentation Network for Face Parsing
Optimized for face parsing with 19 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Handle BatchNorm for small batch sizes in training
        # This condition is for specific cases where batch_size=1 and spatial dimensions are 1x1,
        # which can cause issues with BatchNorm's running statistics.
        # Temporarily switching to eval mode ensures consistent behavior.
        if self.training and x.size(0) == 1 and x.size(2) == 1 and x.size(3) == 1:
            self.bn.eval()
            x = self.bn(x)
            self.bn.train()
        else:
            x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialPath(nn.Module):
    """Spatial Path to preserve spatial information"""
    
    def __init__(self):
        super(SpatialPath, self).__init__()
        # Downsampling convolutions
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        # Final 1x1 convolution to adjust channel count
        self.conv_out = ConvBNReLU(64, 128, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    """ARM: Attention Refinement Module"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionRefinementModule, self).__init__()
        # Convolutional block
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1)
        # 1x1 convolution for attention
        self.conv_attention = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.conv(x)
        # Global average pooling for attention weights
        attention = torch.mean(feature, dim=(2, 3), keepdim=True)
        attention = self.conv_attention(attention)
        attention = self.sigmoid(attention)
        out = feature * attention # Apply attention
        return out


class ContextPath(nn.Module):
    """Context Path with ResNet backbone"""
    
    def __init__(self, backbone: str = 'resnet18'):
        super(ContextPath, self).__init__()
        
        # Import ResNet from torchvision
        import torchvision.models as models
        
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            self.expansion = 1 # BasicBlock does not expand channels
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False)
            self.expansion = 1 # BasicBlock does not expand channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Use ResNet stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # Output 64 channels (1/4 spatial resolution)
        self.layer2 = resnet.layer2 # Output 128 channels (1/8 spatial resolution)
        self.layer3 = resnet.layer3 # Output 256 channels (1/16 spatial resolution)
        self.layer4 = resnet.layer4 # Output 512 channels (1/32 spatial resolution)
        
        # ARM modules: input channels match ResNet stage outputs
        self.arm16 = AttentionRefinementModule(256 * self.expansion, 128) # For 1/16 scale features
        self.arm32 = AttentionRefinementModule(512 * self.expansion, 128) # For 1/32 scale features
        
        # Global average pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBNReLU(512 * self.expansion, 128, kernel_size=1, padding=0)
        
        # Upsampling layers
        self.up16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolution for context path output
        self.conv_out = ConvBNReLU(128, 128, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ResNet backbone forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Get feature maps at different scales
        # feat8 is from layer2, 1/8 resolution, 128 channels
        feat8 = x  
        x = self.layer3(x)
        # feat16 is from layer3, 1/16 resolution, 256 channels
        feat16 = x  
        x = self.layer4(x)
        # feat32 is from layer4, 1/32 resolution, 512 channels
        feat32 = x  
        
        # Global average pooling branch
        gap = self.global_pool(feat32)
        
        # Fix for BatchNorm issue in training mode with batch size 1
        # Similar logic as in ConvBNReLU to handle edge case for BatchNorm
        if self.training and gap.size(0) == 1:
            self.conv_gap.eval()
            gap = self.conv_gap(gap)
            self.conv_gap.train()
        else:
            gap = self.conv_gap(gap)
        
        # ARM processing and feature fusion
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + gap
        feat32_up = self.up32(feat32_sum) # Upsample to 1/16 resolution
        
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum) # Upsample to 1/8 resolution
        
        # Output of Context Path
        context_out = self.conv_out(feat16_up)
        
        # Return context_out (1/8 scale), and original feat16, feat32 for auxiliary heads
        return context_out, feat16, feat32


class FeatureFusionModule(nn.Module):
    """FFM: Feature Fusion Module"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(FeatureFusionModule, self).__init__()
        # 1x1 convolution to combine concatenated features
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Attention branch
        self.conv_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global average pooling
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1), # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1), # Restore channels
            nn.Sigmoid() # Sigmoid for attention weights
        )
    
    def forward(self, spatial_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate features from spatial path and context path
        feat = torch.cat([spatial_feat, context_feat], dim=1)
        feat = self.conv(feat)
        
        # Apply channel attention
        att = self.conv_att(feat)
        feat_att = feat * att
        feat_out = feat + feat_att # Residual connection
        
        return feat_out


class BiSeNet(nn.Module):
    """
    BiSeNet for Face Parsing
    
    Args:
        num_classes: Number of segmentation classes (19 for face parsing)
        backbone: Backbone network ('resnet18' or 'resnet34')
        aux_output: Whether to output auxiliary predictions for deep supervision
    """
    
    def __init__(self, num_classes: int = 19, 
                 backbone: str = 'resnet18',
                 aux_output: bool = True):
        super(BiSeNet, self).__init__()
        
        self.num_classes = num_classes
        self.aux_output = aux_output
        
        # Spatial Path to capture fine-grained details
        self.spatial_path = SpatialPath()
        # Context Path with a backbone for rich contextual information
        self.context_path = ContextPath(backbone)
        
        # Feature Fusion Module to combine spatial and context features
        # Spatial path output is 128 channels, Context path output is 128 channels
        # FFM input is 128 + 128 = 256 channels, output is 256 channels
        self.ffm = FeatureFusionModule(256, 256)
        
        # Final prediction head
        self.conv_out = ConvBNReLU(256, 256, kernel_size=3, padding=1)
        self.conv_pred = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Auxiliary heads for deep supervision during training
        if self.aux_output:
            # Determine input channels for auxiliary heads based on backbone outputs
            # For ResNet18/34:
            # feat16 (output of layer3 in ContextPath) has 256 * expansion channels
            # feat32 (output of layer4 in ContextPath) has 512 * expansion channels
            aux16_in_channels = 256 * self.context_path.expansion
            aux32_in_channels = 512 * self.context_path.expansion

            self.aux_head16 = nn.Sequential(
                ConvBNReLU(aux16_in_channels, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
            self.aux_head32 = nn.Sequential(
                ConvBNReLU(aux32_in_channels, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, num_classes, kernel_size=1)
            )
        
        # Initialize model weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Kaiming Normal for Conv2d and constants for BatchNorm"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use fan_in for more stable initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
        # Special initialization for final prediction layer
        if hasattr(self, 'conv_pred'):
            # Initialize with small values for stable softmax
            nn.init.normal_(self.conv_pred.weight, mean=0, std=0.01)
            if self.conv_pred.bias is not None:
                nn.init.constant_(self.conv_pred.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the BiSeNet model.
        
        Args:
            x: Input tensor (B, 3, H, W) representing an image batch.
        
        Returns:
            Dictionary containing:
                - 'out': Main segmentation output (B, num_classes, H, W).
                - 'aux16': Auxiliary output at 1/16 scale (if aux_output=True and in training mode).
                - 'aux32': Auxiliary output at 1/32 scale (if aux_output=True and in training mode).
        """
        # Get original input size for upsampling final outputs
        h, w = x.size(2), x.size(3)
        
        # Process through Spatial Path
        spatial_out = self.spatial_path(x)
        
        # Process through Context Path to get multi-scale features
        context_out, feat16, feat32 = self.context_path(x)
        
        # Fuse features from Spatial and Context Paths
        fusion_out = self.ffm(spatial_out, context_out)
        
        # Final prediction head
        out = self.conv_out(fusion_out)
        out = self.conv_pred(out)
        # Upsample main output to original input size
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        
        outputs = {'out': out}
        
        # Generate and add auxiliary outputs if enabled and in training mode
        if self.aux_output and self.training:
            aux16 = self.aux_head16(feat16)
            aux16 = F.interpolate(aux16, size=(h, w), mode='bilinear', align_corners=True)
            outputs['aux16'] = aux16
            
            aux32 = self.aux_head32(feat32)
            aux32 = F.interpolate(aux32, size=(h, w), mode='bilinear', align_corners=True)
            outputs['aux32'] = aux32
        
        return outputs
    
    def get_params(self) -> List[Dict]:
        """
        Get parameter groups for differential learning rates.
        Typically, backbone parameters get a lower learning rate.
        """
        # Collect parameters for the backbone (Context Path)
        backbone_params = []
        backbone_params.extend(self.context_path.parameters())
        
        # Collect parameters for the heads (Spatial Path, FFM, and final prediction heads)
        head_params = []
        head_params.extend(self.spatial_path.parameters())
        head_params.extend(self.ffm.parameters())
        head_params.extend(self.conv_out.parameters())
        head_params.extend(self.conv_pred.parameters())
        
        # Add auxiliary head parameters if they exist
        if self.aux_output:
            head_params.extend(self.aux_head16.parameters())
            head_params.extend(self.aux_head32.parameters())
        
        # Return parameter groups with specified learning rate multipliers
        return [
            {'params': backbone_params, 'lr_mult': 0.1}, # Lower LR for backbone
            {'params': head_params, 'lr_mult': 1.0}      # Higher LR for heads
        ]


def create_bisenet(num_classes: int = 19, 
                       backbone: str = 'resnet18',
                       pretrained_path: Optional[str] = None) -> BiSeNet:
        """
        Factory function to create a BiSeNet model with proper pretrained loading.
        """
        model = BiSeNet(num_classes=num_classes, backbone=backbone)
    
        # Load ImageNet pretrained weights for backbone first
        if backbone in ['resnet18', 'resnet34']:
            print(f"Loading ImageNet pretrained weights for {backbone} backbone...")
            import torchvision.models as models
        
            if backbone == 'resnet18':
                pretrained_model = models.resnet18(pretrained=True)
            else:  # resnet34
                pretrained_model = models.resnet34(pretrained=True)
        
            # Load backbone weights
            backbone_state_dict = {}
            for name, param in pretrained_model.named_parameters():
                if name.startswith(('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')):
                    backbone_state_dict[f'context_path.{name}'] = param
        
            # Load with strict=False to allow missing keys
            model.load_state_dict(backbone_state_dict, strict=False)
            print("✅ ImageNet pretrained backbone weights loaded successfully!")
    
        # Load custom pretrained weights if provided
        if pretrained_path is not None:
            print(f"Loading custom pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
        
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
            model.load_state_dict(state_dict, strict=False)
            print("✅ Custom pretrained weights loaded successfully!")
    
        return model


if __name__ == "__main__":
    """Test BiSeNet model functionality when run directly"""
    print("Testing BiSeNet model...")
    
    # Create model with 19 classes (common for face parsing)
    model = create_bisenet(num_classes=19)
    model.eval() # Set to evaluation mode for consistent behavior without training-specific layers
    
    # Define a test input tensor
    batch_size = 2
    height, width = 384, 384
    x = torch.randn(batch_size, 3, height, width)
    
    # Perform a forward pass without gradient computation
    with torch.no_grad():
        outputs = model(x)
    
    # Print shapes of the outputs
    print(f"\nModel outputs:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")
    
    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test parameter grouping for optimizers
    param_groups = model.get_params()
    print(f"\nParameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: {len(list(group['params']))} parameters, lr_mult={group['lr_mult']}")
    
    print("\nBiSeNet test completed!")
