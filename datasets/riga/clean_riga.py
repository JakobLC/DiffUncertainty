from PIL import Image
import numpy as np
from scipy import ndimage as ndi


def label_concentric_rings(
    rings,
    noise_tol=0.05,
    inner_hole_tol=0.05,
    std=1.0,
):
    """
    Returns
    -------
    out : (H, W) ndarray
         0 : outside outer ring
         1 : donut region
         2 : innermost region

    If std == 0, ring/noise pixels remain -1.
    """
    rings = rings.astype(bool)

    free = ~rings
    cc, n_cc = ndi.label(free)

    if n_cc < 3:
        raise ValueError(f"Expected at least 3 non-ring CCs, found {n_cc}")

    ids = np.arange(1, n_cc + 1)
    areas = ndi.sum(np.ones_like(cc, dtype=np.int64), cc, index=ids)

    order = np.argsort(areas)[::-1]
    main_ids = ids[order[:3]]
    extra_ids = ids[order[3:]]

    third_largest_area = areas[order[2]]
    extra_area = areas[order[3:]].sum() if len(order) > 3 else 0

    if extra_area > noise_tol * third_largest_area:
        raise ValueError(
            f"Extra CC area too large: {extra_area} > "
            f"{noise_tol} * {third_largest_area}"
        )

    out = np.full(rings.shape, -1, dtype=np.int8)

    def hole_area(component_id):
        comp = cc == component_id
        filled = ndi.binary_fill_holes(comp)
        return np.count_nonzero(filled & ~comp)

    hole_areas = np.array([hole_area(i) for i in main_ids])
    main_areas = np.array([np.count_nonzero(cc == i) for i in main_ids])

    hole_order = np.argsort(hole_areas)[::-1]

    outside_id = main_ids[hole_order[0]]
    donut_id = main_ids[hole_order[1]]
    center_id = main_ids[hole_order[2]]

    center_hole_area = hole_areas[hole_order[2]]
    center_area = main_areas[hole_order[2]]

    if center_hole_area > inner_hole_tol * center_area:
        raise ValueError(
            f"Innermost CC has too large a hole: {center_hole_area} > "
            f"{inner_hole_tol} * {center_area}"
        )

    out[cc == outside_id] = 0
    out[cc == donut_id] = 1
    out[cc == center_id] = 2

    # Small extra CCs and original ring pixels remain -1 if std == 0
    if std == 0:
        return out

    masks = np.stack(
        [
            cc == outside_id,
            cc == donut_id,
            cc == center_id,
        ],
        axis=0,
    ).astype(float)

    smoothed = np.stack(
        [ndi.gaussian_filter(m, sigma=std) for m in masks],
        axis=0,
    )

    out = np.argmax(smoothed, axis=0).astype(np.int8)

    return out

import argparse
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm


def extract_gt_label(prime_img_path, gt_img_path):
    """
    Extract ground truth label from prime and ground truth images.
    
    Parameters
    ----------
    prime_img_path : str
        Path to the prime (unmodified) image
    gt_img_path : str
        Path to the ground truth (pencil-drawn) image
        
    Returns
    -------
    label : (H, W) ndarray
        Labeled mask with values 0, 1, or 2
    """
    prime_img = np.array(Image.open(prime_img_path))
    gt_img = np.array(Image.open(gt_img_path))
    
    # Compute difference and label concentric rings
    diff = np.abs(prime_img.astype(float) - gt_img.astype(float)).sum(axis=2)
    label = label_concentric_rings(diff > 50)
    
    return label


def find_gt_variants(prime_path):
    """
    Find all 6 ground truth variants for a prime image.
    
    Parameters
    ----------
    prime_path : str
        Path to the prime image
        
    Returns
    -------
    gt_paths : dict
        Dictionary mapping variant number (1-6) to file path, or None if not found
    """
    prime_dir = os.path.dirname(prime_path)
    prime_name = os.path.basename(prime_path)
    
    # Extract base name without extension and "prime"
    if prime_name.endswith('prime.tif'):
        base_name = prime_name[:-9]  # Remove 'prime.tif'
    elif prime_name.endswith('prime.jpg'):
        base_name = prime_name[:-9]  # Remove 'prime.jpg'
    elif prime_name.endswith('prime.tiff'):
        base_name = prime_name[:-10]  # Remove 'prime.tiff'
    elif prime_name.endswith('prime.jpeg'):
        base_name = prime_name[:-10]  # Remove 'prime.jpeg'
    else:
        return None
    
    gt_paths = {}
    
    for variant_num in range(1, 7):
        variant_base = f"{base_name}-{variant_num}"
        
        # Try all possible extensions
        found = False
        for ext in ['.tif', '.tiff', '.jpg', '.jpeg']:
            variant_path = os.path.join(prime_dir, variant_base + ext)
            if os.path.exists(variant_path):
                gt_paths[variant_num] = variant_path
                found = True
                break
        
        # If not found with lowercase, try with capitalized first letter
        if not found:
            capitalized_variant_base = variant_base[0].upper() + variant_base[1:] if variant_base else variant_base
            for ext in ['.tif', '.tiff', '.jpg', '.jpeg']:
                variant_path = os.path.join(prime_dir, capitalized_variant_base + ext)
                if os.path.exists(variant_path):
                    gt_paths[variant_num] = variant_path
                    found = True
                    break
        
        if not found:
            return None  # Missing a variant, can't process this image
    
    return gt_paths


def process_dataset(input_path, output_path=None):
    """
    Process all images in the RIGA dataset and extract ground truth labels.
    
    Parameters
    ----------
    input_path : str
        Root path to search for images
    output_path : str, optional
        Output directory. If None, creates 'cleaned' subfolder in input_path
    """
    input_path = Path(input_path)
    
    # Set default output path
    if output_path is None:
        output_path = input_path / "cleaned"
    else:
        output_path = Path(output_path)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all prime images
    prime_patterns = [
        str(input_path / "**" / "*prime.tif"),
        str(input_path / "**" / "*prime.tiff"),
        str(input_path / "**" / "*prime.jpg"),
        str(input_path / "**" / "*prime.jpeg"),
    ]
    
    prime_files = []
    for pattern in prime_patterns:
        prime_files.extend(glob(pattern, recursive=True))
    
    if not prime_files:
        print(f"No prime images found in {input_path}")
        return
    
    print(f"Found {len(prime_files)} prime images")
    
    seen_names = set()  # Track output names for conflict detection
    processed = 0
    failed = 0
    failed_filenames = []  # Track failed files for final report
    file_counter = 0  # Global counter for file numbering
    
    for prime_path in tqdm(sorted(prime_files), desc="Processing images"):
        try:
            prime_path_obj = Path(prime_path)
            parent_name = prime_path_obj.parent.name
            prime_name = prime_path_obj.stem  # Name without extension
            
            # Clean up filename: remove 'prime' and replace 'image' with 'im'
            prime_name_cleaned = prime_name.replace('prime', '').replace('image', 'im')
            
            # Build output file name: parent_filename_cleaned
            output_name_base = f"{parent_name}_{prime_name_cleaned}"
            
            # Check for naming conflicts
            if output_name_base in seen_names:
                print(f"WARNING: Naming conflict detected for '{output_name_base}'. Skipping.")
                failed_filenames.append(prime_path)
                failed += 1
                continue
            
            seen_names.add(output_name_base)
            
            # Find all ground truth variants
            gt_variants = find_gt_variants(prime_path)
            
            if gt_variants is None:
                print(f"ERROR: Missing ground truth variants for {prime_path}")
                failed_filenames.append(prime_path)
                failed += 1
                continue
            
            # Load prime image and save as PNG
            prime_img = np.array(Image.open(prime_path))
            prime_output_path = output_path / f"{file_counter:03d}_{output_name_base}_prime.png"
            Image.fromarray(prime_img).save(prime_output_path)
            
            # Process each ground truth variant
            for variant_num in range(1, 7):
                try:
                    gt_path = gt_variants[variant_num]
                    label = extract_gt_label(prime_path, gt_path)
                    
                    # Ensure label is uint8 with values 0, 1, 2
                    label_uint8 = label.astype(np.uint8)
                    
                    # Check if label areas are of similar magnitude
                    gt1 = (label_uint8 == 1).sum()
                    gt2 = (label_uint8 == 2).sum()
                    
                    if gt1 < 0.01 * gt2 or gt2 < 0.01 * gt1:
                        print(f"ERROR: Label areas too different for {gt_path} (gt1={gt1}, gt2={gt2})")
                        failed_filenames.append(str(gt_path))
                        failed += 1
                        continue
                    
                    # Verify values are as expected
                    unique_vals = np.unique(label_uint8)
                    if not all(v in [0, 1, 2, 255] for v in unique_vals):  # 255 for -1 (unclassified)
                        print(f"WARNING: Unexpected label values {unique_vals} for {gt_path}")
                    
                    # Save label as PNG with color palette (same prefix as prime image)
                    gt_output_path = output_path / f"{file_counter:03d}_{output_name_base}_gt{variant_num}.png"
                    
                    # Create palette: 0=black, 1=green, 2=red, rest=black
                    palette = [
                        0, 0, 0,        # 0: black
                        0, 255, 0,      # 1: green
                        255, 0, 0,      # 2: red
                    ]
                    # Pad palette to 256 colors (768 bytes)
                    palette.extend([0] * (768 - len(palette)))
                    
                    # Create palette image
                    img_p = Image.fromarray(label_uint8, mode='P')
                    img_p.putpalette(palette)
                    img_p.save(gt_output_path)
                    
                except Exception as e:
                    print(f"ERROR: Failed to process variant {variant_num} for {prime_path}: {e}")
                    failed_filenames.append(str(gt_variants[variant_num]))
                    failed += 1
                    continue
            
            file_counter += 1
            processed += 1
            
        except Exception as e:
            print(f"ERROR: Unexpected failure for {prime_path}: {e}")
            failed_filenames.append(prime_path)
            failed += 1
            continue
    
    print(f"\nProcessing complete: {processed} successful, {failed} failed")
    print(f"Output saved to: {output_path}")
    
    if failed_filenames:
        print("\nFailed files:")
        print("\n".join(failed_filenames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ground truth labels from RIGA retinal images"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/jloch/Desktop/diff/luzern/values_datasets/riga",
        help="Input dataset path (default: %(default)s)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: input_path_cleaned)"
    )
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)