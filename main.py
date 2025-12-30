"""
Color Match Script
Extract color grading from reference image and apply to all input images.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def extract_color_profile(image):
    """
    Extract color profile from reference image in LAB color space.
    Returns mean L, A, B values and standard deviations.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_mean = lab[:, :, 0].mean()
    a_mean = lab[:, :, 1].mean()
    b_mean = lab[:, :, 2].mean()

    l_std = lab[:, :, 0].std()
    a_std = lab[:, :, 1].std()
    b_std = lab[:, :, 2].std()

    return {
        'l_mean': l_mean, 'a_mean': a_mean, 'b_mean': b_mean,
        'l_std': l_std, 'a_std': a_std, 'b_std': b_std
    }


def apply_color_profile(image, profile, strength=1.0, skip_color=False):
    """
    Apply color profile to image.
    Adjusts LAB channels to match reference statistics.

    Args:
        skip_color: If True, only match L (brightness), skip A/B (color) channels
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Get current image stats
    l_mean = lab[:, :, 0].mean()
    a_mean = lab[:, :, 1].mean()
    b_mean = lab[:, :, 2].mean()

    # Calculate adjustments
    l_adjust = (profile['l_mean'] - l_mean) * strength

    # Apply brightness adjustment
    lab[:, :, 0] = np.clip(lab[:, :, 0] + l_adjust, 0, 255)

    # Only apply color matching if not skipping
    if not skip_color:
        a_adjust = (profile['a_mean'] - a_mean) * strength
        b_adjust = (profile['b_mean'] - b_mean) * strength
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_adjust, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_adjust, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def remove_orange(image, hue_min=5, hue_max=25, blur_size=11, chroma_strength=0.85):
    """
    Remove orange tones by desaturating them in HSV space.
    This reduces orange energy instead of spreading it.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Detect orange in HSV (lighting invariant)
    mask_orange = cv2.inRange(h.astype(np.uint8), hue_min, hue_max)
    mask_orange = cv2.GaussianBlur(mask_orange, (blur_size, blur_size), 0)
    alpha = mask_orange.astype(np.float32) / 255.0

    # Desaturate orange areas only (reduce saturation)
    s = s * (1 - alpha * chroma_strength)

    hsv = cv2.merge((h, np.clip(s, 0, 255), v))
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Match colors from reference image to input images"
    )
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Reference image path (the look you want)"
    )
    parser.add_argument(
        "--input", "-i",
        default="input",
        help="Input directory with images to process (default: input)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--strength", "-s",
        type=float,
        default=1.0,
        help="Color match strength 0.0-1.0 (default: 1.0 = full match)"
    )
    # Orange removal parameters
    parser.add_argument(
        "--remove-orange",
        action="store_true",
        help="Enable orange removal from background"
    )
    parser.add_argument(
        "--orange-hue-min",
        type=int,
        default=5,
        help="Orange hue range min (default: 5)"
    )
    parser.add_argument(
        "--orange-hue-max",
        type=int,
        default=25,
        help="Orange hue range max (default: 25)"
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=11,
        help="Gaussian blur kernel size (default: 11)"
    )
    parser.add_argument(
        "--chroma-strength",
        type=float,
        default=0.85,
        help="Chroma normalization strength (default: 0.85)"
    )

    args = parser.parse_args()

    # Load reference image
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"❌ Reference image not found: {ref_path}")
        return

    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        print(f"❌ Failed to load reference: {ref_path}")
        return

    print(f"📊 Extracting color profile from: {ref_path.name}")
    profile = extract_color_profile(ref_img)
    print(f"   L: {profile['l_mean']:.1f}±{profile['l_std']:.1f}")
    print(f"   A: {profile['a_mean']:.1f}±{profile['a_std']:.1f}")
    print(f"   B: {profile['b_mean']:.1f}±{profile['b_std']:.1f}\n")

    # Setup directories
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    input_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not input_files:
        print(f"❌ No images found in: {input_path}")
        return

    print(f"🎨 Processing {len(input_files)} images...\n")

    # Process each input
    for idx, input_file in enumerate(input_files, 1):
        print(f"[{idx}/{len(input_files)}] {input_file.name}", end=" ")

        img = cv2.imread(str(input_file))
        if img is None:
            print("❌ Failed to load")
            continue

        # Step 1: Remove orange FIRST (before color matching)
        if args.remove_orange:
            img = remove_orange(
                img,
                hue_min=args.orange_hue_min,
                hue_max=args.orange_hue_max,
                blur_size=args.blur_size,
                chroma_strength=args.chroma_strength
            )

        # Step 2: Apply color matching
        # Skip A/B color channels if orange removal is enabled
        skip_color = args.remove_orange
        result = apply_color_profile(img, profile, strength=args.strength, skip_color=skip_color)

        # Save output
        output_file = output_path / input_file.name
        cv2.imwrite(str(output_file), result)
        print("✅")

    print(f"\n✅ Done! Saved to: {output_path}")


if __name__ == "__main__":
    main()
