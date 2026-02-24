
"""
Given a path inside the save folder e.g. (specified as local, just set default to "chaksu128")
"{ROOT_DIR}/saves/chaksu128/"
the script should look for all saved models (subfolders not called ..saves/chaksu128/test_results/) and see
if hparams.yaml exists. If it exists it should be loaded and if the logger is a wandb logger:
logger:
   _target_: pytorch_lightning.loggers.WandbLogger
then the script should make some adjustments to the hparams file.
first, replace the logger with a TensorBoard logger:
logger:
  _target_: pytorch_lightning.loggers.TensorBoardLogger
  save_dir: ${save_dir}
  name: ${exp_name}
  default_hp_metric: false
Next, replace the save dir with:
save_dir: {ROOT_DIR}/saves

Make identical changes to the hparams key within [PARENT OF hparams.yaml]/checkpoints/last*.ckpt.
Infact, the changes should be made to the ckpt file first, since the hparams.yaml file is the 
identifier for whether the ckpt file should be changed or not.

The script should have a --dry param which if set to true, should only print the identified hparams.yaml
files and the changes that would be made, without actually making any changes. The script should also have a
tqdm bar to show progress over how many of the wandb identified hparams.yaml files have been processed. 
The script should also print a summary at the end of how many hparams.yaml files were identified, 
and how many were changed. Make sure only wandb logger hparams.yaml files are changed.
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
import glob

ROOT_DIR = "/home/jloch/Desktop/diff/luzern/values"

def is_wandb_logger(hparams):
    """Check if the logger in hparams is a WandbLogger."""
    if 'logger' not in hparams:
        return False
    logger = hparams['logger']
    if isinstance(logger, dict) and '_target_' in logger:
        return 'WandbLogger' in logger['_target_']
    return False


def update_hparams(hparams):
    """Update hparams to replace WandbLogger with TensorBoardLogger and fix save_dir."""
    modified = False
    
    # Replace logger
    if 'logger' in hparams and is_wandb_logger(hparams):
        hparams['logger'] = {
            '_target_': 'pytorch_lightning.loggers.TensorBoardLogger',
            'save_dir': '${save_dir}',
            'name': '${exp_name}',
            'default_hp_metric': False
        }
        modified = True
    
    # Replace save_dir
    if 'save_dir' in hparams:
        hparams['save_dir'] = f'{ROOT_DIR}/saves'
        modified = True
    
    return hparams, modified


def process_checkpoint(ckpt_path, dry_run=False, raise_error=False):
    """Process a checkpoint file to update its hparams.
    Returns (success, modified) where success=False means an error occurred
    (hparams.yaml should not be updated), and modified=True means the ckpt was changed.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'hyper_parameters' not in checkpoint:
            return True, False
        
        hparams = checkpoint['hyper_parameters']
        
        if not is_wandb_logger(hparams):
            # Already updated or never had wandb - not an error, safe to proceed
            return True, False
        
        updated_hparams, modified = update_hparams(hparams)
        
        if modified and not dry_run:
            checkpoint['hyper_parameters'] = updated_hparams
            torch.save(checkpoint, ckpt_path)
        
        return True, modified
    except Exception as e:
        if raise_error:
            raise
        print(f"Error processing checkpoint {ckpt_path}: {e}")
        return False, False


def process_hparams_yaml(yaml_path, dry_run=False, raise_error=False):
    """Process a hparams.yaml file to update logger and save_dir."""
    try:
        with open(yaml_path, 'r') as f:
            hparams = yaml.safe_load(f)
        
        if not is_wandb_logger(hparams):
            return False
        
        updated_hparams, modified = update_hparams(hparams)
        
        if modified and not dry_run:
            with open(yaml_path, 'w') as f:
                yaml.dump(updated_hparams, f, default_flow_style=False, sort_keys=False)
        
        return modified
    except Exception as e:
        if raise_error:
            raise
        print(f"Error processing hparams.yaml {yaml_path}: {e}")
        return False


def find_and_process_models(base_path, dry_run=False, raise_error=False):
    """Find all model directories and process their hparams files."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist!")
        return
    
    # Find all hparams.yaml files, excluding test_results directories
    all_hparams = []
    for hparams_file in base_path.rglob('hparams.yaml'):
        # Skip if in test_results directory
        if 'test_results' in hparams_file.parts:
            continue
        all_hparams.append(hparams_file)
    
    # Filter to only wandb logger hparams
    wandb_hparams = []
    for hparams_file in all_hparams:
        try:
            with open(hparams_file, 'r') as f:
                hparams = yaml.safe_load(f)
            if is_wandb_logger(hparams):
                wandb_hparams.append(hparams_file)
        except:
            continue
    
    print(f"Found {len(all_hparams)} total hparams.yaml files")
    print(f"Found {len(wandb_hparams)} wandb logger hparams.yaml files")
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
    
    changed_count = 0
    
    for hparams_file in tqdm(wandb_hparams, desc="Processing models"):
        model_dir = hparams_file.parent
        
        if dry_run:
            print(f"\nWould process: {str(hparams_file).replace(ROOT_DIR,'${ROOT_DIR}')}")
        
        # First, process checkpoint files
        checkpoints_dir = model_dir / 'checkpoints'
        ckpt_modified = False
        
        ckpt_success = True
        if checkpoints_dir.exists():
            ckpt_files = glob.glob(str(checkpoints_dir / 'last*.ckpt'))
            for ckpt_file in ckpt_files:
                if dry_run:
                    print(f"  Update: {str(ckpt_file).replace(ROOT_DIR,'${ROOT_DIR}')}")
                success, modified = process_checkpoint(ckpt_file, dry_run, raise_error=raise_error)
                if not success:
                    ckpt_success = False
                if modified:
                    ckpt_modified = True
        
        # Only process hparams.yaml if all checkpoints were processed without error (or in dry run mode)
        if ckpt_success or dry_run:
            yaml_modified = process_hparams_yaml(hparams_file, dry_run, raise_error=raise_error)
            
            if ckpt_modified or yaml_modified:
                changed_count += 1
                if dry_run:
                    print(f"  Update: {str(hparams_file).replace(ROOT_DIR,'${ROOT_DIR}')}")
        else:
            print(f"  Skipping hparams.yaml due to checkpoint error: {hparams_file}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total hparams.yaml files identified: {len(wandb_hparams)}")
    print(f"Files {'that would be' if dry_run else ''} changed: {changed_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Fix wandb logger configs in saved models'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='chaksu128',
        help='Dataset folder name (default: chaksu128)'
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run - only show what would be changed without making changes'
    )
    parser.add_argument(
        '--raise_error',
        action='store_true',
        help='Raise errors instead of printing and continuing'
    )
    
    args = parser.parse_args()
    
    base_path = f'{ROOT_DIR}/saves/{args.dataset}'
    
    find_and_process_models(base_path, dry_run=args.dry, raise_error=args.raise_error)


if __name__ == '__main__':
    main()