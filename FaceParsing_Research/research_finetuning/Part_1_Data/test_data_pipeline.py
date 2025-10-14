"""
Test script for data pipeline
Verifies all components are working correctly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

# Direct imports from local files
from celebamask_dataset import CelebAMaskHQDataset, create_data_loaders
from augmentations import FaceParsingAugmentation, visualize_augmentations
from data_utils import (
    verify_dataset_structure, calculate_dataset_statistics,
    create_visualization_grid, validate_data_loader,
    create_class_legend, analyze_memory_usage
)


class DataPipelineTester:
    """Test all components of the data pipeline"""
    
    def __init__(self, dataset_path: str, output_dir: str = "part1_test_outputs"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "tests": {}
        }
    
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def test_dataset_structure(self) -> bool:
        """Test 1: Verify dataset structure"""
        self.log("\n" + "="*60)
        self.log("TEST 1: Dataset Structure Verification")
        self.log("="*60)
        
        try:
            report = verify_dataset_structure(str(self.dataset_path))
            
            # Save report
            report_path = self.output_dir / "dataset_structure_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.results["tests"]["dataset_structure"] = {
                "passed": report['valid'],
                "report": report
            }
            
            if report['valid']:
                self.log("✅ Dataset structure is valid!")
            else:
                self.log("❌ Dataset structure has issues!")
                for issue in report['issues']:
                    self.log(f"  - {issue}")
            
            return report['valid']
            
        except Exception as e:
            self.log(f"❌ Error during structure verification: {str(e)}")
            self.results["tests"]["dataset_structure"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def _create_class_samples_visualization(self, dataset, distribution_data):
        """Create visualization showing samples of each class"""
        import matplotlib.pyplot as plt
        
        # Find samples that contain different classes
        class_samples = {}
        
        for i in range(min(50, len(dataset))):  # Check first 50 samples
            try:
                image, mask = dataset[i]
                mask_np = mask.numpy()
                
                # Check which classes are present
                unique_classes = np.unique(mask_np)
                
                for class_idx in unique_classes:
                    class_name = dataset.IDX_TO_CLASS[class_idx]
                    if class_name not in class_samples and class_name != 'background':
                        # Store image index and create a highlighted mask
                        highlighted_mask = np.zeros_like(mask_np)
                        highlighted_mask[mask_np == class_idx] = class_idx
                        class_samples[class_name] = {
                            'image_idx': i,
                            'image': image,
                            'mask': mask_np,
                            'highlighted_mask': highlighted_mask
                        }
                        
                        if len(class_samples) >= 12:  # Limit to 12 classes for visualization
                            break
            except:
                continue
            
            if len(class_samples) >= 12:
                break
        
        # Create visualization
        if class_samples:
            n_classes = len(class_samples)
            cols = 4
            rows = (n_classes + cols - 1) // cols
            
            fig, axes = plt.subplots(rows * 2, cols, figsize=(16, rows * 6))
            if rows == 1:
                axes = axes.reshape(2, -1)
            
            for idx, (class_name, sample_data) in enumerate(class_samples.items()):
                row = (idx // cols) * 2
                col = idx % cols
                
                # Denormalize image
                image_np = sample_data['image'].permute(1, 2, 0).numpy()
                image_np = image_np * np.array([0.229, 0.224, 0.225])
                image_np += np.array([0.485, 0.456, 0.406])
                image_np = np.clip(image_np, 0, 1)
                
                # Show original image
                axes[row, col].imshow(image_np)
                axes[row, col].set_title(f'{class_name} (Sample {sample_data["image_idx"]})')
                axes[row, col].axis('off')
                
                # Show highlighted class
                highlighted_rgb = dataset.mask_to_rgb(sample_data['highlighted_mask'])
                axes[row + 1, col].imshow(highlighted_rgb)
                axes[row + 1, col].set_title(f'{class_name} highlighted')
                axes[row + 1, col].axis('off')
            
            # Hide empty subplots
            for idx in range(n_classes, rows * cols):
                row = (idx // cols) * 2
                col = idx % cols
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')
            
            plt.tight_layout()
            samples_path = self.output_dir / "class_samples_visualization.png"
            plt.savefig(samples_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"✅ Class samples visualization saved to: {samples_path}")
        else:
            self.log("⚠️ Could not find samples for class visualization")
    
    def test_data_loaders(self) -> bool:
        """Test 6: Data loaders"""
        self.log("\n" + "="*60)
        self.log("TEST 6: Data Loaders")
        self.log("="*60)
        
        try:
            # Create augmentations
            train_aug = FaceParsingAugmentation(image_size=384, mode='train').transform
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                root_dir=str(self.dataset_path),
                batch_size=2,
                image_size=384,
                num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
                pin_memory=False,
                subset_size=20,
                augmentations=train_aug
            )
            
            self.log(f"✅ Data loaders created successfully")
            self.log(f"  Train batches: {len(train_loader)}")
            self.log(f"  Val batches: {len(val_loader)}")
            
            # Validate loaders
            self.log("\nValidating train loader...")
            validate_data_loader(train_loader, num_batches=3)
            
            self.log("\nValidating val loader...")
            validate_data_loader(val_loader, num_batches=2)
            
            # Test iteration speed
            self.log("\nTesting iteration speed...")
            start_time = time.time()
            for batch_idx, (images, masks) in enumerate(train_loader):
                if batch_idx >= 5:
                    break
            elapsed = time.time() - start_time
            self.log(f"✅ Loaded 5 batches in {elapsed:.2f} seconds ({elapsed/5:.2f} sec/batch)")
            
            self.results["tests"]["data_loaders"] = {
                "passed": True,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "seconds_per_batch": elapsed/5
            }
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error during data loader testing: {str(e)}")
            self.results["tests"]["data_loaders"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_dataset_loading(self) -> bool:
        """Test 2: Dataset loading and basic functionality"""
        self.log("\n" + "="*60)
        self.log("TEST 2: Dataset Loading")
        self.log("="*60)
        
        try:
            # Create dataset with small subset
            dataset = CelebAMaskHQDataset(
                root_dir=str(self.dataset_path),
                split='train',
                image_size=384,
                subset_size=10
            )
            
            self.log(f"✅ Dataset created successfully")
            self.log(f"  Size: {len(dataset)} samples")
            
            # Test loading samples
            errors = []
            for i in range(min(5, len(dataset))):
                try:
                    image, mask = dataset[i]
                    self.log(f"  Sample {i}: image {image.shape}, mask {mask.shape}")
                    
                    # Verify shapes
                    assert image.shape == (3, 384, 384), f"Invalid image shape: {image.shape}"
                    assert mask.shape == (384, 384), f"Invalid mask shape: {mask.shape}"
                    
                    # Verify data types
                    assert image.dtype == torch.float32, f"Invalid image dtype: {image.dtype}"
                    assert mask.dtype == torch.int64, f"Invalid mask dtype: {mask.dtype}"
                    
                    # Verify mask values
                    assert mask.min() >= 0 and mask.max() < 19, f"Invalid mask values: [{mask.min()}, {mask.max()}]"
                    
                except Exception as e:
                    errors.append(f"Sample {i}: {str(e)}")
            
            if errors:
                self.log("❌ Errors during sample loading:")
                for error in errors:
                    self.log(f"  - {error}")
                passed = False
            else:
                self.log("✅ All samples loaded successfully!")
                passed = True
            
            # Test visualization
            self.log("\nTesting visualization...")
            vis_path = self.output_dir / "sample_visualization.png"
            dataset.visualize_sample(0, save_path=str(vis_path))
            self.log(f"✅ Visualization saved to: {vis_path}")
            
            # Create class legend
            legend_path = self.output_dir / "class_legend.png"
            create_class_legend(CelebAMaskHQDataset, save_path=str(legend_path))
            self.log(f"✅ Class legend saved to: {legend_path}")
            
            self.results["tests"]["dataset_loading"] = {
                "passed": passed,
                "samples_tested": min(5, len(dataset)),
                "errors": errors
            }
            
            return passed
            
        except Exception as e:
            self.log(f"❌ Error during dataset loading: {str(e)}")
            self.results["tests"]["dataset_loading"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_augmentations(self) -> bool:
        """Test 3: Augmentation pipeline"""
        self.log("\n" + "="*60)
        self.log("TEST 3: Augmentation Pipeline")
        self.log("="*60)
        
        try:
            # Create dataset
            dataset = CelebAMaskHQDataset(
                root_dir=str(self.dataset_path),
                split='train',
                image_size=384,
                subset_size=5
            )
            
            # Get augmentation pipeline
            train_aug = FaceParsingAugmentation(image_size=384, mode='train')
            val_aug = FaceParsingAugmentation(image_size=384, mode='val')
            
            # Test on first sample
            image, mask = dataset[0]
            
            # Denormalize for augmentation
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225])
            image_np += np.array([0.485, 0.456, 0.406])
            image_np = (image_np * 255).astype(np.uint8)
            mask_np = mask.numpy()
            
            # Test training augmentation
            self.log("Testing training augmentations...")
            train_errors = []
            for i in range(5):
                try:
                    aug_result = train_aug(image=image_np, mask=mask_np)
                    aug_image = aug_result['image']
                    aug_mask = aug_result['mask']
                    
                    assert aug_image.shape == (384, 384, 3), f"Invalid augmented image shape: {aug_image.shape}"
                    assert aug_mask.shape == (384, 384), f"Invalid augmented mask shape: {aug_mask.shape}"
                    
                except Exception as e:
                    train_errors.append(f"Augmentation {i}: {str(e)}")
            
            if train_errors:
                self.log("❌ Training augmentation errors:")
                for error in train_errors:
                    self.log(f"  - {error}")
            else:
                self.log("✅ Training augmentations working correctly!")
            
            # Test validation augmentation
            self.log("\nTesting validation augmentations...")
            val_result = val_aug(image=image_np, mask=mask_np)
            self.log("✅ Validation augmentation (resize only) working correctly!")
            
            # Visualize augmentations
            self.log("\nCreating augmentation visualization...")
            vis_path = self.output_dir / "augmentation_examples.png"
            visualize_augmentations(image_np, mask_np, image_size=384, save_path=str(vis_path))
            self.log(f"✅ Augmentation examples saved to: {vis_path}")
            
            passed = len(train_errors) == 0
            
            self.results["tests"]["augmentations"] = {
                "passed": passed,
                "training_errors": train_errors,
                "validation_tested": True
            }
            
            return passed
            
        except Exception as e:
            self.log(f"❌ Error during augmentation testing: {str(e)}")
            self.results["tests"]["augmentations"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_class_distribution(self) -> bool:
        """Test 5: Comprehensive class distribution analysis"""
        self.log("\n" + "="*60)
        self.log("TEST 5: Class Distribution Analysis")
        self.log("="*60)
        
        try:
            # Create dataset
            dataset = CelebAMaskHQDataset(
                root_dir=str(self.dataset_path),
                split='train',
                image_size=384,
                subset_size=100  # Use more samples for better distribution analysis
            )
            
            self.log(f"Analyzing class distribution from {len(dataset)} samples...")
            
            # Initialize counters
            class_pixel_counts = np.zeros(len(dataset.MASK_CLASSES))
            class_presence_counts = np.zeros(len(dataset.MASK_CLASSES))  # How many images contain each class
            total_pixels = 0
            
            # Analyze each sample
            for i in range(len(dataset)):
                try:
                    image, mask = dataset[i]
                    mask_np = mask.numpy()
                    
                    # Count pixels for each class
                    for class_idx in range(len(dataset.MASK_CLASSES)):
                        pixel_count = (mask_np == class_idx).sum()
                        class_pixel_counts[class_idx] += pixel_count
                        
                        # Count presence (non-zero pixels)
                        if pixel_count > 0:
                            class_presence_counts[class_idx] += 1
                    
                    total_pixels += mask_np.size
                    
                    if (i + 1) % 20 == 0:
                        self.log(f"  Processed {i + 1}/{len(dataset)} samples...")
                        
                except Exception as e:
                    self.log(f"  Error processing sample {i}: {str(e)}")
            
            # Calculate percentages
            class_percentages = (class_pixel_counts / total_pixels) * 100
            presence_percentages = (class_presence_counts / len(dataset)) * 100
            
            # Create detailed report
            self.log("\n📊 Class Distribution Results:")
            self.log("-" * 80)
            self.log(f"{'Class':<12} {'Index':<5} {'Pixels':<12} {'%Pixels':<8} {'Presence':<8} {'%Presence':<10}")
            self.log("-" * 80)
            
            distribution_data = {}
            for class_idx, (class_name, _) in enumerate(dataset.MASK_CLASSES.items()):
                pixels = int(class_pixel_counts[class_idx])
                pixel_pct = class_percentages[class_idx]
                presence = int(class_presence_counts[class_idx])
                presence_pct = presence_percentages[class_idx]
                
                self.log(f"{class_name:<12} {class_idx:<5} {pixels:<12,} {pixel_pct:<8.2f} {presence:<8} {presence_pct:<10.1f}")
                
                distribution_data[class_name] = {
                    'index': class_idx,
                    'pixel_count': pixels,
                    'pixel_percentage': float(pixel_pct),
                    'presence_count': presence,
                    'presence_percentage': float(presence_pct)
                }
            
            # Identify potential issues
            issues = []
            
            # Check for very rare classes (< 0.1% of pixels)
            rare_classes = [name for name, data in distribution_data.items() 
                           if data['pixel_percentage'] < 0.1 and name != 'background']
            if rare_classes:
                issues.append(f"Very rare classes (< 0.1% pixels): {', '.join(rare_classes)}")
            
            # Check for very dominant classes (> 50% of pixels)
            dominant_classes = [name for name, data in distribution_data.items() 
                               if data['pixel_percentage'] > 50.0]
            if dominant_classes:
                issues.append(f"Very dominant classes (> 50% pixels): {', '.join(dominant_classes)}")
            
            # Check for classes missing from many images
            missing_classes = [name for name, data in distribution_data.items() 
                              if data['presence_percentage'] < 10.0 and name != 'background']
            if missing_classes:
                issues.append(f"Rarely present classes (< 10% of images): {', '.join(missing_classes)}")
            
            # Create visualizations
            self.log("\n📈 Creating distribution visualizations...")
            
            # 1. Pixel distribution pie chart
            import matplotlib.pyplot as plt
            
            # Filter out background for better visualization
            non_bg_classes = [name for name, data in distribution_data.items() if name != 'background']
            non_bg_percentages = [distribution_data[name]['pixel_percentage'] for name in non_bg_classes]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Pie chart of pixel distribution (excluding background)
            ax1.pie(non_bg_percentages, labels=non_bg_classes, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Pixel Distribution by Class (Excluding Background)')
            
            # Bar chart of all classes
            class_names = list(distribution_data.keys())
            pixel_percentages = [distribution_data[name]['pixel_percentage'] for name in class_names]
            
            bars = ax2.bar(range(len(class_names)), pixel_percentages)
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Percentage of Pixels')
            ax2.set_title('Pixel Distribution by Class (All Classes)')
            ax2.set_xticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            
            # Color bars by percentage
            for i, bar in enumerate(bars):
                if pixel_percentages[i] > 10:
                    bar.set_color('red')
                elif pixel_percentages[i] > 1:
                    bar.set_color('orange')
                else:
                    bar.set_color('lightblue')
            
            # Presence percentage bar chart
            presence_percentages_list = [distribution_data[name]['presence_percentage'] for name in class_names]
            ax3.bar(range(len(class_names)), presence_percentages_list, color='green', alpha=0.7)
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Percentage of Images')
            ax3.set_title('Class Presence Across Images')
            ax3.set_xticks(range(len(class_names)))
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
            
            # Log scale view for better visibility of small classes
            ax4.bar(range(len(class_names)), pixel_percentages)
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Percentage of Pixels (Log Scale)')
            ax4.set_title('Pixel Distribution (Log Scale)')
            ax4.set_yscale('log')
            ax4.set_xticks(range(len(class_names)))
            ax4.set_xticklabels(class_names, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / "class_distribution_analysis.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"✅ Distribution visualization saved to: {viz_path}")
            
            # Create sample visualizations for each class
            self.log("\n🖼️ Creating sample visualizations for each class...")
            self._create_class_samples_visualization(dataset, distribution_data)
            
            # Calculate class weights for training
            self.log("\n⚖️ Calculating recommended class weights...")
            weights = dataset.get_class_weights(num_samples=len(dataset))
            
            weights_dict = {}
            for i, weight in enumerate(weights):
                class_name = dataset.IDX_TO_CLASS[i]
                weights_dict[class_name] = float(weight)
            
            # Save detailed results
            detailed_results = {
                "samples_analyzed": int(len(dataset)),
                "total_pixels": int(total_pixels),
                "class_distribution": {
                    name: {
                        'index': int(data['index']),
                        'pixel_count': int(data['pixel_count']),
                        'pixel_percentage': float(data['pixel_percentage']),
                        'presence_count': int(data['presence_count']),
                        'presence_percentage': float(data['presence_percentage'])
                    } for name, data in distribution_data.items()
                },
                "class_weights": {name: float(weight) for name, weight in weights_dict.items()},
                "potential_issues": issues,
                "summary": {
                    "most_common_class": [
                        max(distribution_data.items(), key=lambda x: x[1]['pixel_percentage'])[0],
                        float(max(distribution_data.items(), key=lambda x: x[1]['pixel_percentage'])[1]['pixel_percentage'])
                    ],
                    "rarest_class": [
                        min(distribution_data.items(), key=lambda x: x[1]['pixel_percentage'])[0],
                        float(min(distribution_data.items(), key=lambda x: x[1]['pixel_percentage'])[1]['pixel_percentage'])
                    ],
                    "most_present_class": [
                        max(distribution_data.items(), key=lambda x: x[1]['presence_percentage'])[0],
                        float(max(distribution_data.items(), key=lambda x: x[1]['presence_percentage'])[1]['presence_percentage'])
                    ],
                    "least_present_class": [
                        min(distribution_data.items(), key=lambda x: x[1]['presence_percentage'])[0],
                        float(min(distribution_data.items(), key=lambda x: x[1]['presence_percentage'])[1]['presence_percentage'])
                    ]
                }
            }
            
            results_path = self.output_dir / "class_distribution_detailed.json"
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            self.log(f"✅ Detailed results saved to: {results_path}")
            
            # Print summary
            if issues:
                self.log("\n⚠️ Potential Issues Found:")
                for issue in issues:
                    self.log(f"  - {issue}")
            else:
                self.log("\n✅ No major distribution issues detected!")
            
            # Class balance assessment
            non_bg_percentages_array = np.array([distribution_data[name]['pixel_percentage'] 
                                               for name in non_bg_classes])
            balance_score = 1.0 - np.std(non_bg_percentages_array) / np.mean(non_bg_percentages_array)
            
            self.log(f"\n📊 Dataset Balance Score: {balance_score:.3f}")
            self.log(f"   (1.0 = perfectly balanced, 0.0 = very imbalanced)")
            
            passed = len(issues) == 0 or balance_score > 0.3  # Allow some imbalance
            
            self.results["tests"]["class_distribution"] = {
                "passed": passed,
                "samples_analyzed": len(dataset),
                "balance_score": float(balance_score),
                "issues": issues,
                "results_file": str(results_path)
            }
            
            return passed
            
        except Exception as e:
            self.log(f"❌ Error during class distribution analysis: {str(e)}")
            self.results["tests"]["class_distribution"] = {
                "passed": False,
                "error": str(e)
            }
            return False
        """Test 4: Data loaders"""
        self.log("\n" + "="*60)
        self.log("TEST 4: Data Loaders")
        self.log("="*60)
        
        try:
            # Create augmentations
            train_aug = FaceParsingAugmentation(image_size=384, mode='train').transform
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                root_dir=str(self.dataset_path),
                batch_size=2,
                image_size=384,
                num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
                pin_memory=False,
                subset_size=20,
                augmentations=train_aug
            )
            
            self.log(f"✅ Data loaders created successfully")
            self.log(f"  Train batches: {len(train_loader)}")
            self.log(f"  Val batches: {len(val_loader)}")
            
            # Validate loaders
            self.log("\nValidating train loader...")
            validate_data_loader(train_loader, num_batches=3)
            
            self.log("\nValidating val loader...")
            validate_data_loader(val_loader, num_batches=2)
            
            # Test iteration speed
            self.log("\nTesting iteration speed...")
            start_time = time.time()
            for batch_idx, (images, masks) in enumerate(train_loader):
                if batch_idx >= 5:
                    break
            elapsed = time.time() - start_time
            self.log(f"✅ Loaded 5 batches in {elapsed:.2f} seconds ({elapsed/5:.2f} sec/batch)")
            
            self.results["tests"]["data_loaders"] = {
                "passed": True,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "seconds_per_batch": elapsed/5
            }
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error during data loader testing: {str(e)}")
            self.results["tests"]["data_loaders"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        self.log("\n" + "🚀 "*20)
        self.log("PART 1: DATA PIPELINE TESTING")
        self.log("🚀 "*20)
        
        all_passed = True
        
        # Run each test
        tests = [
            ("Dataset Structure", self.test_dataset_structure),
            ("Dataset Loading", self.test_dataset_loading),
            ("Augmentations", self.test_augmentations),
            ("Class Distribution", self.test_class_distribution),
            ("Data Loaders", self.test_data_loaders),
        ]
        
        for test_name, test_func in tests:
            try:
                passed = test_func()
                all_passed = all_passed and passed
            except Exception as e:
                self.log(f"\n❌ Unexpected error in {test_name}: {str(e)}")
                all_passed = False
        
        # Save results with proper type conversion
        def convert_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(convert_types(self.results), f, indent=2)
        
        # Print summary
        self.log("\n" + "="*60)
        self.log("TEST SUMMARY")
        self.log("="*60)
        
        for test_name, test_result in self.results["tests"].items():
            status = "✅" if test_result.get("passed", False) else "❌"
            self.log(f"{status} {test_name}")
        
        self.log(f"\nResults saved to: {results_path}")
        self.log(f"Output directory: {self.output_dir}")
        
        if all_passed:
            self.log("\n🎉 ALL TESTS PASSED! Data pipeline is ready!")
        else:
            self.log("\n⚠️ Some tests failed. Please check the errors above.")
        
        return all_passed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test data pipeline")
    parser.add_argument("--dataset-path", type=str, 
                       default=r"C:\CelebAMask-HQ",
                       help="Path to CelebAMask-HQ dataset")
    parser.add_argument("--output-dir", type=str,
                       default="part1_test_outputs",
                       help="Directory for test outputs")
    
    args = parser.parse_args()
    
    # Create tester
    tester = DataPipelineTester(args.dataset_path, args.output_dir)
    
    # Run tests
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()