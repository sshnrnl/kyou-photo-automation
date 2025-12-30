"""
Color Match Streamlit App
Batch image processing with reference image color matching
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
import time
import shutil

# Configuration
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
REFERENCES_DIR = Path("references")

st.set_page_config(
    page_title="Color Match Tool",
    page_icon="🎨",
    layout="wide"
)

# Create directories
for folder in [REFERENCES_DIR, INPUT_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COLOR MATCHING FUNCTIONS (from main.py)
# =============================================================================

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


def process_image(img, profile, strength=1.0, remove_orange_enabled=False,
                  orange_hue_min=5, orange_hue_max=25, blur_size=11, chroma_strength=0.85):
    """
    Process a single image with color matching and optional orange removal.
    """
    # Step 1: Remove orange FIRST (before color matching)
    if remove_orange_enabled:
        img = remove_orange(
            img,
            hue_min=orange_hue_min,
            hue_max=orange_hue_max,
            blur_size=blur_size,
            chroma_strength=chroma_strength
        )

    # Step 2: Apply color matching
    # Skip A/B color channels if orange removal is enabled
    skip_color = remove_orange_enabled
    result = apply_color_profile(img, profile, strength=strength, skip_color=skip_color)

    return result


# =============================================================================
# REFERENCE IMAGE MANAGEMENT
# =============================================================================

def get_all_references():
    """Get list of all saved reference images"""
    references = []
    for file in REFERENCES_DIR.glob("*_profile.json"):
        references.append(file.stem.replace("_profile", ""))
    return sorted(references)


def save_reference(name, img, profile):
    """Save reference image and profile"""
    ref_path = REFERENCES_DIR / f"{name}.jpg"
    profile_path = REFERENCES_DIR / f"{name}_profile.json"

    cv2.imwrite(str(ref_path), img)
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)
    return True


def load_reference_profile(name):
    """Load reference profile from file"""
    profile_path = REFERENCES_DIR / f"{name}_profile.json"
    if profile_path.exists():
        with open(profile_path, 'r') as f:
            return json.load(f)
    return None


def load_reference_image(name):
    """Load reference image from file"""
    ref_path = REFERENCES_DIR / f"{name}.jpg"
    if ref_path.exists():
        return cv2.imread(str(ref_path))
    return None


def delete_reference(name):
    """Delete reference image and profile"""
    ref_path = REFERENCES_DIR / f"{name}.jpg"
    profile_path = REFERENCES_DIR / f"{name}_profile.json"

    if ref_path.exists():
        ref_path.unlink()
    if profile_path.exists():
        profile_path.unlink()
    return True


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("🎨 Color Match Tool")
    st.markdown("*Batch image processing with reference color matching*")
    st.markdown("---")

    # Initialize session state
    if 'current_reference' not in st.session_state:
        st.session_state.current_reference = None
        st.session_state.current_profile = None

    # Sidebar - Reference Management
    with st.sidebar:
        st.header("📎 Reference Image")

        # Tab for upload or saved references
        ref_tab1, ref_tab2 = st.tabs(["Upload", "Saved"])

        with ref_tab1:
            st.info("Upload a reference image to extract its color grading")

            reference_upload = st.file_uploader(
                "Upload reference",
                type=["jpg", "jpeg", "png", "bmp", "webp", "jfif"],
                key="ref_upload"
            )

            if reference_upload:
                # Read and display reference image
                ref_bytes = np.asarray(bytearray(reference_upload.read()), dtype=np.uint8)
                ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)

                if ref_img is not None:
                    # Extract LAB statistics
                    ref_profile = extract_color_profile(ref_img)

                    # Display reference image
                    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    st.image(ref_rgb, caption="Reference Preview", width=250)

                    # Show LAB stats
                    st.caption(f"L: {ref_profile['l_mean']:.1f}±{ref_profile['l_std']:.1f}")
                    st.caption(f"A: {ref_profile['a_mean']:.1f}±{ref_profile['a_std']:.1f}")
                    st.caption(f"B: {ref_profile['b_mean']:.1f}±{ref_profile['b_std']:.1f}")

                    # Save option
                    save_name = st.text_input("Save as:", placeholder="my_reference", key="save_ref_name")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save", key="save_ref_btn") and save_name:
                            save_reference(save_name, ref_img, ref_profile)
                            st.success(f"Saved: {save_name}")
                            time.sleep(0.5)
                            st.rerun()
                    with col2:
                        if st.button("✅ Use", key="use_ref_btn"):
                            st.session_state.current_reference = reference_upload.name
                            st.session_state.current_profile = ref_profile
                            st.success("Reference set!")
                            time.sleep(0.5)
                            st.rerun()

        with ref_tab2:
            saved_references = get_all_references()

            if saved_references:
                selected_ref = st.selectbox("Select Reference", saved_references, key="select_ref")

                if selected_ref:
                    # Load and display preview
                    ref_img = load_reference_image(selected_ref)
                    ref_profile = load_reference_profile(selected_ref)

                    if ref_img is not None:
                        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                        st.image(ref_rgb, caption=selected_ref, width=200)

                        if ref_profile:
                            st.caption(f"L: {ref_profile['l_mean']:.1f}±{ref_profile['l_std']:.1f}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("📥 Load", key="load_ref_btn"):
                                st.session_state.current_reference = selected_ref
                                st.session_state.current_profile = ref_profile
                                st.success(f"Loaded: {selected_ref}")
                                time.sleep(0.5)
                                st.rerun()
                        with col2:
                            if st.button("🗑️ Delete", key="del_ref_btn"):
                                delete_reference(selected_ref)
                                if st.session_state.current_reference == selected_ref:
                                    st.session_state.current_reference = None
                                    st.session_state.current_profile = None
                                st.success(f"Deleted: {selected_ref}")
                                time.sleep(0.5)
                                st.rerun()
                        with col3:
                            if st.button("👁️ View", key="view_ref_btn"):
                                st.json(ref_profile)
            else:
                st.info("No saved references")

        # Show current reference
        st.markdown("---")
        if st.session_state.current_reference:
            st.success(f"🎨 Current: **{st.session_state.current_reference}**")
        else:
            st.warning("⚠️ No reference selected")

        # Parameters
        st.markdown("---")
        st.header("⚙️ Parameters")

        strength = st.slider(
            "Match Strength",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="How strongly to apply the color match"
        )

        st.markdown("---")
        st.subheader("🍊 Orange Removal")

        remove_orange_enabled = st.checkbox(
            "Enable Orange Removal",
            value=False,
            help="Remove orange tones BEFORE color matching"
        )

        if remove_orange_enabled:
            st.caption("Note: When enabled, only brightness (L) is matched")

            col1, col2 = st.columns(2)
            with col1:
                orange_hue_min = st.slider("Hue Min", 0, 25, 5, 1)
            with col2:
                orange_hue_max = st.slider("Hue Max", 5, 50, 25, 1)

            blur_size = st.slider("Blur Size", 3, 31, 11, 2)
            chroma_strength = st.slider("Chroma Strength", 0.0, 1.0, 0.85, 0.05)
        else:
            orange_hue_min = 5
            orange_hue_max = 25
            blur_size = 11
            chroma_strength = 0.85

        # Actions
        st.markdown("---")
        st.header("🗑️ Actions")

        if st.button("🧹 Clear Outputs", key="clear_outputs"):
            if OUTPUT_DIR.exists():
                shutil.rmtree(OUTPUT_DIR)
                st.success("Cleared output folder")
                st.rerun()

        if st.button("🗑️ Clear Inputs", key="clear_inputs"):
            if INPUT_DIR.exists():
                shutil.rmtree(INPUT_DIR)
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                st.success("Cleared input folder")
                st.rerun()

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("📤 Upload Images")

        uploaded_files = st.file_uploader(
            "Choose images to process",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "jfif"],
            accept_multiple_files=True,
            key="upload_files"
        )

    with col2:
        st.header("📁 Input Folder")
        if INPUT_DIR.exists():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".jfif"}
            input_count = len([f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in image_extensions])
            st.info(f"**{input_count}** images in input/ folder")
        else:
            st.info("input/ folder is empty")

    # Process uploaded files
    if uploaded_files:
        st.markdown("---")
        st.write(f"**{len(uploaded_files)}** image(s) selected")

        # Check if reference is selected
        if not st.session_state.current_profile:
            st.warning("⚠️ Please select a reference image first!")
        else:
            if st.button("🚀 Process Uploaded Images", type="primary", key="process_uploads"):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Save to input folder
                    temp_path = INPUT_DIR / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Process image
                    img = cv2.imread(str(temp_path))
                    if img is not None:
                        result = process_image(
                            img,
                            st.session_state.current_profile,
                            strength=strength,
                            remove_orange_enabled=remove_orange_enabled,
                            orange_hue_min=orange_hue_min,
                            orange_hue_max=orange_hue_max,
                            blur_size=blur_size,
                            chroma_strength=chroma_strength
                        )
                        cv2.imwrite(str(OUTPUT_DIR / uploaded_file.name), result)

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.text("✅ Done!")
                st.success(f"Processed **{len(uploaded_files)}** images!")
                st.balloons()

    # Process input folder
    elif INPUT_DIR.exists():
        st.markdown("---")
        st.header("📁 Process Input Folder")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".jfif"}
        image_files = [f for f in INPUT_DIR.glob("*.*") if f.suffix.lower() in image_extensions]

        if image_files:
            st.write(f"Found **{len(image_files)}** image(s) in `input/` folder")

            # Check if reference is selected
            if not st.session_state.current_profile:
                st.warning("⚠️ Please select a reference image first!")
            else:
                if st.button("🚀 Process All Images", type="primary", key="process_folder"):
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, img_file in enumerate(image_files):
                        status_text.text(f"Processing {img_file.name}...")

                        img = cv2.imread(str(img_file))
                        if img is not None:
                            result = process_image(
                                img,
                                st.session_state.current_profile,
                                strength=strength,
                                remove_orange_enabled=remove_orange_enabled,
                                orange_hue_min=orange_hue_min,
                                orange_hue_max=orange_hue_max,
                                blur_size=blur_size,
                                chroma_strength=chroma_strength
                            )
                            cv2.imwrite(str(OUTPUT_DIR / img_file.name), result)

                        progress_bar.progress((idx + 1) / len(image_files))

                    status_text.text("✅ Done!")
                    st.success(f"Processed **{len(image_files)}** images!")
                    st.balloons()

    # Results preview
    st.markdown("---")
    st.header("🖼️ Output Preview")

    if OUTPUT_DIR.exists():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".jfif"}
        output_images = [f for f in OUTPUT_DIR.glob("*.*") if f.suffix.lower() in image_extensions]

        if output_images:
            st.write(f"**{len(output_images)}** processed image(s)")
            cols = st.columns(min(4, len(output_images)))
            for idx, img_file in enumerate(output_images[:8]):
                with cols[idx % 4]:
                    st.image(str(img_file), caption=img_file.name, use_container_width=True)
        else:
            st.info("No processed images yet. Upload a reference and some images to get started!")
    else:
        st.info("No processed images yet. Upload a reference and some images to get started!")


if __name__ == "__main__":
    main()
