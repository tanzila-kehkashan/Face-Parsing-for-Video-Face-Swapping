#!/usr/bin/env python3
"""
Optimal 5K CelebAMask-HQ Split Creator
Selects best 5,000 images and creates optimal train/val/test splits for maximum performance
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set, Optional
import random
from collections import defaultdict, Counter
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

class OptimalImageSelector:
    """Selects the best 5,000 images for optimal training performance"""
    
    def __init__(self, root_dir: str):
        self.root_path = Path(root_dir)
        self.image_dir = self.root_path / 'CelebA-HQ-img'
        self.mask_dir = self.root_path / 'CelebAMask-HQ-mask-anno'
        self.attr_file = self.root_path / 'CelebAMask-HQ-attribute-anno.txt'
        
        # All possible mask classes for completeness scoring
        self.mask_classes = [
            'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g',
            'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 
            'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
        ]
        
        # Core classes that should be prioritized
        self.core_classes = ['skin', 'hair', 'l_eye', 'r_eye', 'nose', 'mouth', 'l_brow', 'r_brow']
    
    def select_optimal_images(self, target_count: int = 5000) -> List[int]:
        """Select the best images for training performance"""
        print(f"🎯 Selecting optimal {target_count:,} images for maximum training performance...")
        
        # Step 1: Find all available images
        available_images = self._find_available_images()
        if len(available_images) < target_count:
            print(f"⚠️  Only {len(available_images)} images available, using all")
            return sorted(available_images)
        
        print(f"📊 Found {len(available_images):,} total images")
        
        # Step 2: Score images based on mask completeness and quality
        image_scores = self._score_images(available_images)
        
        # Step 3: Load attributes for diversity if available
        attr_df = self._load_attributes()
        
        # Step 4: Select optimal subset with diversity
        selected_images = self._select_diverse_subset(image_scores, attr_df, target_count)
        
        print(f"✅ Selected {len(selected_images):,} optimal images")
        return sorted(selected_images)
    
    def _find_available_images(self) -> List[int]:
        """Find all images with valid IDs"""
        image_files = list(self.image_dir.glob('*.jpg'))
        valid_ids = []
        
        for img_file in image_files:
            try:
                img_id = int(img_file.stem)
                if img_id >= 0:
                    valid_ids.append(img_id)
            except ValueError:
                continue
        
        return sorted(valid_ids)
    
    def _score_images(self, image_ids: List[int]) -> Dict[int, float]:
        """Score images based on mask completeness and quality"""
        print("📊 Scoring images based on mask completeness...")
        
        image_scores = {}
        
        for img_id in tqdm(image_ids, desc="Scoring images"):
            score = self._calculate_image_score(img_id)
            if score > 0:  # Only include images with some masks
                image_scores[img_id] = score
        
        print(f"✅ Scored {len(image_scores):,} images with valid masks")
        return image_scores
    
    def _calculate_image_score(self, img_id: int) -> float:
        """Calculate quality score for a single image"""
        # Find correct subdirectory
        folder_idx = img_id // 2000
        mask_folder = self.mask_dir / str(folder_idx)
        
        if not mask_folder.exists():
            return 0.0
        
        score = 0.0
        img_id_padded = f"{img_id:05d}"
        
        # Score based on mask completeness
        core_masks_found = 0
        total_masks_found = 0
        
        for class_name in self.mask_classes:
            mask_file = mask_folder / f"{img_id_padded}_{class_name}.png"
            if mask_file.exists():
                total_masks_found += 1
                if class_name in self.core_classes:
                    core_masks_found += 1
                    score += 2.0  # Core classes weighted higher
                else:
                    score += 1.0  # Additional classes
        
        # Bonus for having most core classes
        core_completeness = core_masks_found / len(self.core_classes)
        if core_completeness >= 0.8:  # 80% of core classes
            score += 5.0
        elif core_completeness >= 0.6:  # 60% of core classes
            score += 2.0
        
        # Bonus for overall completeness
        total_completeness = total_masks_found / len(self.mask_classes)
        score += total_completeness * 3.0
        
        return score
    
    def _load_attributes(self) -> Optional[pd.DataFrame]:
        """Load attribute annotations for diversity selection"""
        if not self.attr_file.exists():
            print("⚠️  No attributes file found, using mask-based selection only")
            return None
        
        try:
            print("📋 Loading attributes for diversity selection...")
            with open(self.attr_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 2:
                attr_df = pd.read_csv(self.attr_file, sep=r'\s+', skiprows=1, header=0)
                first_col = attr_df.columns[0]
                attr_df.rename(columns={first_col: 'image_id'}, inplace=True)
                attr_df['image_id'] = attr_df['image_id'].astype(int)
                print(f"✅ Loaded attributes for {len(attr_df)} images")
                return attr_df
        except Exception as e:
            print(f"⚠️  Could not load attributes: {e}")
        
        return None
    
    def _select_diverse_subset(self, image_scores: Dict[int, float], attr_df: Optional[pd.DataFrame], target_count: int) -> List[int]:
        """Select diverse subset of high-quality images"""
        print(f"🎯 Selecting diverse subset of {target_count:,} high-quality images...")
        
        # Get top candidates (select more than needed for diversity filtering)
        candidates_count = min(target_count * 3, len(image_scores))
        top_candidates = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)[:candidates_count]
        candidate_ids = [img_id for img_id, _ in top_candidates]
        
        print(f"📊 Top {len(candidate_ids):,} candidates selected for diversity filtering")
        
        if attr_df is None:
            # No attributes available, just take top scored images
            return candidate_ids[:target_count]
        
        # Select diverse subset using attributes
        return self._select_with_attribute_diversity(candidate_ids, attr_df, target_count)
    
    def _select_with_attribute_diversity(self, candidate_ids: List[int], attr_df: pd.DataFrame, target_count: int) -> List[int]:
        """Select subset with good attribute diversity"""
        print("🌈 Applying diversity selection based on attributes...")
        
        # Key attributes for diversity
        diversity_attrs = []
        for attr in ['Male', 'Young', 'Smiling', 'Wearing_Earrings', 'Eyeglasses', 'Mustache', 'Goatee', 'Bald']:
            if attr in attr_df.columns:
                diversity_attrs.append(attr)
        
        if not diversity_attrs:
            print("⚠️  No suitable diversity attributes found")
            return candidate_ids[:target_count]
        
        print(f"📊 Using diversity attributes: {diversity_attrs[:4]}")  # Show first 4
        
        # Filter candidates to those with attributes
        attr_subset = attr_df[attr_df['image_id'].isin(candidate_ids)]
        available_with_attrs = list(attr_subset['image_id'])
        
        if len(available_with_attrs) < target_count:
            # Not enough with attributes, fill with remaining candidates
            remaining = [img_id for img_id in candidate_ids if img_id not in available_with_attrs]
            selected = available_with_attrs + remaining[:target_count - len(available_with_attrs)]
            return selected[:target_count]
        
        # Create balanced selection across attribute combinations
        selected_images = []
        attr_combinations = {}
        
        # Group by key attribute combinations (first 2 attributes)
        key_attrs = diversity_attrs[:2]
        for _, row in attr_subset.iterrows():
            img_id = int(row['image_id'])
            key = tuple(int(row[attr]) for attr in key_attrs)
            if key not in attr_combinations:
                attr_combinations[key] = []
            attr_combinations[key].append(img_id)
        
        # Select proportionally from each group
        total_combinations = len(attr_combinations)
        per_combination = target_count // total_combinations
        remainder = target_count % total_combinations
        
        for i, (combination, img_ids) in enumerate(attr_combinations.items()):
            # Determine how many to select from this group
            to_select = per_combination
            if i < remainder:
                to_select += 1
            
            # Select best images from this group (they're already scored)
            group_selection = img_ids[:min(to_select, len(img_ids))]
            selected_images.extend(group_selection)
        
        # Fill remaining spots if needed
        if len(selected_images) < target_count:
            remaining_candidates = [img_id for img_id in candidate_ids if img_id not in selected_images]
            selected_images.extend(remaining_candidates[:target_count - len(selected_images)])
        
        print(f"✅ Selected {len(selected_images)} diverse, high-quality images")
        return selected_images[:target_count]

def create_optimal_splits(selected_images: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Create optimal train/val/test splits from selected images"""
    print(f"📊 Creating optimal splits from {len(selected_images):,} selected images...")
    
    # Optimal ratios for 5K dataset
    train_ratio = 0.80  # 4,000 images
    val_ratio = 0.15    # 750 images  
    test_ratio = 0.05   # 250 images
    
    # Shuffle for randomness
    images = selected_images.copy()
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_split = sorted(images[:train_end])
    val_split = sorted(images[train_end:val_end])
    test_split = sorted(images[val_end:])
    
    print(f"📋 Split results:")
    print(f"   Train: {len(train_split):,} images ({len(train_split)/n:.1%})")
    print(f"   Val:   {len(val_split):,} images ({len(val_split)/n:.1%})")
    print(f"   Test:  {len(test_split):,} images ({len(test_split)/n:.1%})")
    
    return train_split, val_split, test_split

def save_splits(train_ids: List[int], val_ids: List[int], test_ids: List[int], output_dir: str) -> Dict[str, str]:
    """Save optimal split files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train.txt': train_ids,
        'val.txt': val_ids,
        'test.txt': test_ids
    }
    
    saved_files = {}
    
    for filename, ids in splits.items():
        file_path = output_path / filename
        
        with open(file_path, 'w') as f:
            for img_id in ids:
                f.write(f"{img_id}\n")
        
        print(f"✅ Saved {filename}: {len(ids):,} images")
        saved_files[filename] = str(file_path)
    
    return saved_files

def main():
    """Main function for optimal 5K split creation"""
    parser = argparse.ArgumentParser(description="Create optimal 5K CelebAMask-HQ splits for maximum performance")
    parser.add_argument("--dataset-root", type=str, default="/content/CelebAMask-HQ",
                       help="Root directory of CelebAMask-HQ dataset")
    parser.add_argument("--output-dir", type=str, default="/content/CelebAMask-HQ/splits",
                       help="Output directory for split files")
    parser.add_argument("--target-images", type=int, default=5000,
                       help="Target number of images to select (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print(f"🎯 OPTIMAL {args.target_images:,} IMAGE SPLIT CREATOR FOR CELEBAMASK-HQ")
    print("=" * 80)
    
    # Step 1: Select optimal images
    print(f"\n📋 STEP 1: Selecting Optimal {args.target_images:,} Images")
    selector = OptimalImageSelector(args.dataset_root)
    selected_images = selector.select_optimal_images(args.target_images)
    
    if len(selected_images) < 1000:
        print(f"❌ Error: Only {len(selected_images)} quality images found (minimum 1000 required)")
        sys.exit(1)
    
    # Step 2: Create optimal splits
    print(f"\n📋 STEP 2: Creating Optimal Splits")
    train_ids, val_ids, test_ids = create_optimal_splits(selected_images)
    
    # Step 3: Save splits
    print(f"\n📋 STEP 3: Saving Split Files")
    saved_files = save_splits(train_ids, val_ids, test_ids, args.output_dir)
    
    # Create summary
    summary = {
        'dataset_root': str(args.dataset_root),
        'selection_method': 'optimal_mask_completeness_with_diversity',
        'target_images': args.target_images,
        'actual_images': len(selected_images),
        'splits': {
            'train': {'count': len(train_ids), 'ratio': len(train_ids)/len(selected_images)},
            'val': {'count': len(val_ids), 'ratio': len(val_ids)/len(selected_images)},
            'test': {'count': len(test_ids), 'ratio': len(test_ids)/len(selected_images)}
        },
        'files': saved_files,
        'parameters': vars(args)
    }
    
    summary_file = Path(args.output_dir) / "optimal_splits_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved: {summary_file}")
    
    # Final success message
    print("\n" + "=" * 80)
    print("🎉 OPTIMAL 5K SPLITS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"✅ Selected {len(selected_images):,} highest-quality images")
    print(f"✅ Train split: {len(train_ids):,} images (high-quality, diverse)")
    print(f"✅ Val split: {len(val_ids):,} images (balanced validation)")
    print(f"✅ Test split: {len(test_ids):,} images (representative testing)")
    print(f"✅ All files saved to: {args.output_dir}")
    print(f"\n🚀 Ready for optimal training performance!")

if __name__ == "__main__":
    main()