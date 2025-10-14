"""
Test suite for model components
Tests BiSeNet, LoRA layers, and model builder functionality
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from research_finetuning.Part_2_Model.bisenet import BiSeNet, create_bisenet
from research_finetuning.Part_2_Model.lora_layers import (
    LoRAConv2d, LoRALinear, apply_lora_to_model
)
from research_finetuning.Part_2_Model.model_builder import (
    BiSeNetLoRA, build_model_from_config
)
from research_finetuning.Part_2_Model.model_utils import (
    count_parameters, analyze_model_memory, profile_model_speed
)


class TestBiSeNet:
    """Test BiSeNet model"""
    
    def test_model_creation(self):
        """Test model can be created"""
        model = create_bisenet(num_classes=19)
        assert isinstance(model, BiSeNet)
        assert model.num_classes == 19
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = create_bisenet(num_classes=19)
        model.eval()
        
        x = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            outputs = model(x)
        
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        assert outputs['out'].shape == (2, 19, 384, 384)
    
    def test_training_mode(self):
        """Test model in training mode with auxiliary outputs"""
        model = create_bisenet(num_classes=19, backbone='resnet18')
        model.train() # Set model to training mode to enable auxiliary outputs
        
        x = torch.randn(2, 3, 256, 256)  # Changed to batch size 2 to avoid BatchNorm issues
        outputs = model(x)
        
        assert 'out' in outputs
        assert 'aux16' in outputs
        assert 'aux32' in outputs
        # Verify shapes of auxiliary outputs
        assert outputs['aux16'].shape == (2, 19, 256, 256)
        assert outputs['aux32'].shape == (2, 19, 256, 256)
    
    def test_different_input_sizes(self):
        """Test model with different input sizes"""
        model = create_bisenet(num_classes=19)
        model.eval()
        
        sizes = [(256, 256), (384, 384), (512, 512)]
        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                outputs = model(x)
            assert outputs['out'].shape == (1, 19, h, w)


class TestLoRALayers:
    """Test LoRA layer implementations"""
    
    def test_lora_conv2d(self):
        """Test LoRAConv2d layer"""
        # Create base conv layer
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        original_weight = conv.weight.data.clone()
        
        # Wrap with LoRA
        lora_conv = LoRAConv2d(conv, rank=4, alpha=8)
        
        # Check weight freezing
        assert not conv.weight.requires_grad
        assert lora_conv.lora_A.requires_grad
        assert lora_conv.lora_B.requires_grad
        
        # Test forward pass
        x = torch.randn(2, 64, 32, 32)
        out = lora_conv(x)
        assert out.shape == (2, 128, 32, 32)
        
        # Check original weights unchanged (LoRA adds to output, doesn't modify base weights directly until merge)
        assert torch.allclose(conv.weight.data, original_weight)
    
    def test_lora_linear(self):
        """Test LoRALinear layer"""
        linear = nn.Linear(128, 256)
        lora_linear = LoRALinear(linear, rank=8, alpha=16)
        
        x = torch.randn(4, 128)
        out = lora_linear(x)
        assert out.shape == (4, 256)
    
    def test_apply_lora_to_model(self):
        """Test applying LoRA to entire model"""
        model = create_bisenet(num_classes=19)
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply LoRA
        model, stats = apply_lora_to_model(model, rank=4, alpha=8)
        
        # Check stats
        assert stats['lora_modules'] > 0
        assert stats['trainable_params'] < original_params
        assert stats['trainable_params'] == stats['lora_params']
        
        # Check that non-LoRA params are frozen
        for name, param in model.named_parameters():
            if 'lora_' in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad
    
    def test_merge_weights(self):
        """Test merging LoRA weights"""
        conv = nn.Conv2d(32, 64, kernel_size=3)
        lora_conv = LoRAConv2d(conv, rank=4)
        
        # Get output before merge
        x = torch.randn(1, 32, 16, 16)
        lora_conv.eval() # Set to eval mode to ensure LoRA is applied correctly for comparison
        out_before = lora_conv(x)
        
        # Merge weights
        lora_conv.merge_weights()
        # After merging, LoRA is disabled, so the forward pass uses the modified base conv
        out_after = lora_conv(x)
        
        # Outputs should be very close after merging (due to floating point precision)
        assert torch.allclose(out_before, out_after, atol=1e-6)


class TestModelBuilder:
    """Test model builder functionality"""
    
    def test_bisenet_lora_creation(self):
        """Test BiSeNetLoRA creation"""
        model = BiSeNetLoRA(
            num_classes=19,
            backbone='resnet18',
            lora_config={'rank': 4, 'alpha': 8}
        )
        
        assert hasattr(model, 'model')
        assert hasattr(model, 'lora_config')
        assert hasattr(model, 'lora_stats')
    
    def test_lora_parameter_extraction(self):
        """Test getting LoRA parameters"""
        model = BiSeNetLoRA(num_classes=19)
        lora_params = model.get_lora_parameters()
        
        assert len(lora_params) > 0
        
        # Get IDs of lora_params for efficient identity comparison
        lora_param_ids = {id(p) for p in lora_params}

        # Check that all extracted parameters are indeed LoRA parameters and are trainable
        found_lora_param_names = []
        for name, param in model.model.named_parameters():
            if id(param) in lora_param_ids: # Compare object identity using id()
                found_lora_param_names.append(name)
                assert 'lora_' in name # Ensure the name contains 'lora_'
                assert param.requires_grad # Ensure it's trainable
            else:
                # Ensure non-LoRA parameters are not in the list and are not trainable
                assert 'lora_' not in name or not param.requires_grad
        
        # Ensure that the number of found LoRA parameters matches the extracted list
        assert len(found_lora_param_names) == len(lora_params)
    
    def test_parameter_groups(self):
        """Test parameter group creation"""
        model = BiSeNetLoRA(num_classes=19)
        param_groups = model.get_parameter_groups(base_lr=1e-4)
        
        assert len(param_groups) >= 1
        assert param_groups[0]['name'] == 'lora'
        assert param_groups[0]['lr'] == 1e-4
        
        # Ensure that all trainable parameters are included in some group
        all_trainable_params = {id(p) for p in model.parameters() if p.requires_grad}
        grouped_params = {id(p) for group in param_groups for p in group['params']}
        assert all_trainable_params == grouped_params
    
    def test_save_load_lora_weights(self):
        """Test saving and loading LoRA weights"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model and save weights
            model1 = BiSeNetLoRA(num_classes=19)
            save_path = Path(tmpdir) / "lora_weights.pth"
            model1.save_lora_weights(save_path)
            
            # Create new model and load weights
            model2 = BiSeNetLoRA(num_classes=19)
            model2.load_lora_weights(save_path)
            
            # Compare LoRA parameters by iterating through named parameters
            # and checking only those that are LoRA specific and trainable.
            lora_params1 = {name: param for name, param in model1.model.named_parameters() if 'lora_' in name and param.requires_grad}
            lora_params2 = {name: param for name, param in model2.model.named_parameters() if 'lora_' in name and param.requires_grad}

            assert lora_params1.keys() == lora_params2.keys()
            for name in lora_params1.keys():
                assert torch.allclose(lora_params1[name], lora_params2[name])
    
    def test_build_from_config(self):
        """Test building model from config"""
        config = {
            'model': {
                'num_classes': 19,
                'backbone': 'resnet18',
                'lora_rank': 8,
                'lora_alpha': 16,
                'lora_exclude_modules': ['conv1']
            }
        }
        
        model = build_model_from_config(config)
        assert isinstance(model, BiSeNetLoRA)
        assert model.lora_config['rank'] == 8
        assert model.lora_config['alpha'] == 16
        assert 'conv1' in model.lora_config['exclude_modules']


class TestModelUtils:
    """Test model utility functions"""
    
    def test_count_parameters(self):
        """Test parameter counting"""
        model = create_bisenet(num_classes=19)
        param_counts = count_parameters(model)
        
        assert 'Total' in param_counts
        assert param_counts['Total'] > 0
        assert 'Conv2d' in param_counts
    
    def test_memory_analysis(self):
        """Test memory analysis"""
        model = create_bisenet(num_classes=19)
        memory_stats = analyze_model_memory(
            model, 
            input_shape=(1, 3, 256, 256),
            device='cpu'
        )
        
        assert 'param_memory_mb' in memory_stats
        assert 'forward_memory_mb' in memory_stats
        assert all(v >= 0 for v in memory_stats.values())
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_speed_cuda(self):
        """Test speed profiling on CUDA"""
        model = create_bisenet(num_classes=19)
        speed_stats = profile_model_speed(
            model,
            input_shape=(1, 3, 384, 384),
            num_runs=10,
            device='cuda'
        )
        
        assert 'mean_ms' in speed_stats
        assert 'fps' in speed_stats
        assert speed_stats['fps'] > 0


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, '-v'])
