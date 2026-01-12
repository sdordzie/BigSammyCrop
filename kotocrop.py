"""
-------------------------------------------------------------------------------
APP NAME: BigSammyCrop
TAGLINE: Smart African Precision Cropping
AUTHOR: Samuel Kwame Dordzie (BigSammy Graphics Consult)
TYPE: Single-File Streamlit Application
-------------------------------------------------------------------------------
"""

# =============================================================================
# 1. IMPORTS & GLOBAL CONFIGURATION
# =============================================================================
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import io
import zipfile
import os
import gc  # Garbage Collector for memory management
from datetime import datetime

# --- TRY IMPORTING STREAMLIT-CROPPER ---
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except ImportError:
    HAS_CROPPER = False

# --- TRY IMPORTING REMBG (BACKGROUND REMOVER) ---
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# Set page config immediately
st.set_page_config(
    page_title="BigSammyCrop | Smart Precision",
    page_icon="🦀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. UI STYLING & STEALTH MODE
# =============================================================================
def hide_streamlit_elements():
    """
    Hides the Streamlit hamburger menu, footer, 'Deploy' button, and 'Stop' button.
    """
    hide_st_style = """
        <style>
        /* 1. Hide the hamburger menu (top right) */
        #MainMenu {visibility: hidden;}
        
        /* 2. Hide the "Made with Streamlit" footer */
        footer {visibility: hidden;}
        
        /* 3. Hide the top header line (colored decoration) */
        header {visibility: hidden;}
        
        /* 4. Hide the "Deploy" and "Stop" buttons */
        .stDeployButton {display:none;}
        [data-testid="stStatusWidget"] {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

def apply_ghanaian_theme():
    """
    Injects custom CSS to style the app with Ghanaian colors while respecting
    the user's Light/Dark mode settings.
    """
    st.markdown("""
        <style>
        /* --- 1. GHANAIAN PALETTE --- */
        :root {
            --gh-red: #CE1126;
            --gh-gold: #FCD116;
            --gh-green: #006B3F;
            --gh-black: #000000;
        }

        /* --- 2. ADAPTIVE HEADER --- */
        header[data-testid="stHeader"] {
            border-bottom: 5px solid var(--gh-gold);
        }
        
        /* --- 3. ACCENT LINES --- */
        .decoration-line {
            height: 4px;
            width: 100%;
            background: linear-gradient(90deg, var(--gh-red) 33%, var(--gh-gold) 33%, var(--gh-gold) 66%, var(--gh-green) 66%);
            margin-bottom: 20px;
            border-radius: 2px;
        }

        /* --- 4. BUTTONS --- */
        div.stButton > button {
            background-color: var(--gh-black);
            color: white !important;
            border-radius: 6px;
            border: 1px solid #333;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: var(--gh-green);
            color: white !important;
            border: 1px solid var(--gh-gold);
        }

        /* --- 5. SIDEBAR ADAPTATION --- */
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--gh-gold);
        }
        
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: var(--text-color) !important;
        }

        /* --- 6. FILE UPLOADER (ADAPTIVE) --- */
        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed var(--text-color) !important;
        }
        
        /* --- 7. TYPOGRAPHY --- */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            border: 1px solid #888;
        }
        </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 3. CROP PRESETS
# =============================================================================
PRESETS = {
    "Custom Dimensions": (0, 0),
    "Instagram Square (1:1)": (1080, 1080),
    "Instagram Story (9:16)": (1080, 1920),
    "WhatsApp DP": (500, 500),
    "YouTube Thumbnail (16:9)": (1280, 720),
    "A4 Print (300 DPI)": (2480, 3508),
    "Ghana Passport (35x45mm)": (413, 531),  
    "Ghana School ID": (300, 375),
    "Event Flyer (Square)": (2000, 2000)
}

# =============================================================================
# 4. UTILITY FUNCTIONS (CACHED FOR STABILITY)
# =============================================================================
@st.cache_data
def load_image_from_bytes(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return ImageOps.exif_transpose(image) 
    except Exception as e:
        return None

def pil_to_cv2(pil_image):
    pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

# =============================================================================
# 5. BACKGROUND ENGINE (CLOUD SAFE MODE)
# =============================================================================
class BackgroundEngine:
    @staticmethod
    def process_background(img, mode, custom_color="#FFFFFF"):
        if mode == "Original" or not HAS_REMBG:
            return img
        
        try:
            # Use 'u2netp' (lightweight) model to prevent cloud crashes
            img_no_bg = rembg_remove(img, model_name="u2netp")
        except Exception:
            try:
                # Fallback to standard if lightweight fails
                img_no_bg = rembg_remove(img)
            except Exception:
                return img
        
        if mode == "Transparent":
            return img_no_bg
        
        if mode == "White": bg_color = (255, 255, 255)
        elif mode == "Blue": bg_color = (0, 120, 215) 
        elif mode == "Red": bg_color = (200, 16, 46) 
        elif mode == "Custom":
            h = custom_color.lstrip('#')
            bg_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        else:
            return img_no_bg

        new_bg = Image.new("RGBA", img_no_bg.size, bg_color + (255,))
        new_bg.paste(img_no_bg, (0, 0), img_no_bg)
        
        return new_bg.convert("RGB")

@st.cache_data
def get_processed_image(file_bytes, brightness, contrast, sharpness, bg_mode, bg_custom):
    img = load_image_from_bytes(file_bytes)
    if img is None: return None
    
    if bg_mode != "Original":
        img = BackgroundEngine.process_background(img, bg_mode, bg_custom)
        
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
    return img

# =============================================================================
# 6. AI INTELLIGENCE ENGINE (UPDATED FOR SENSITIVITY)
# =============================================================================
class IntelligenceEngine:
    def __init__(self):
        self.face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

    def detect_face_rect(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # Optimized for smaller faces / full body shots
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) == 0: return None
        return max(faces, key=lambda r: r[2] * r[3])

# =============================================================================
# 7. CROP ENGINE (UPDATED WITH SAFE FALLBACK)
# =============================================================================
class CropEngine:
    @staticmethod
    def stabilize_image(cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        if lines is not None:
            angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]
            valid_angles = [a for a in angles if -10 < a < 10]
            if valid_angles:
                median_angle = np.median(valid_angles)
                (h, w) = cv_img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
                return cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return cv_img

    @staticmethod
    def smart_crop(pil_img, target_w, target_h, ai_engine, use_ai=True, debug_mode=False):
        cv_img = pil_to_cv2(pil_img)
        img_h, img_w = cv_img.shape[:2]
        target_ratio = target_w / target_h
        current_ratio = img_w / img_h
        
        if current_ratio > target_ratio:
            base_crop_h = img_h
            base_crop_w = int(base_crop_h * target_ratio)
        else:
            base_crop_w = img_w
            base_crop_h = int(base_crop_w / target_ratio)
            
        detected_face = None
        crop_w, crop_h = base_crop_w, base_crop_h
        
        if use_ai:
            detected_face = ai_engine.detect_face_rect(cv_img)
            
            if detected_face is not None:
                # --- CASE 1: FACE FOUND ---
                fx, fy, fw, fh = detected_face
                ideal_crop_h = int(fh * 2.3)
                if 100 < ideal_crop_h < img_h:
                    crop_h = min(ideal_crop_h, base_crop_h)
                    crop_w = int(crop_h * target_ratio)
                    if crop_w > img_w:
                        crop_w = img_w
                        crop_h = int(crop_w / target_ratio)
                center_x = fx + (fw // 2)
                shift_amount = int(crop_h * 0.08)
                center_y = (fy + (fh // 2)) + shift_amount
            else:
                # --- CASE 2: NO FACE FOUND (SAFE FALLBACK) ---
                # Fixes the issue where it crops shoes. Defaults to upper body.
                center_x = img_w // 2
                center_y = int(img_h * 0.35) 
        else:
            center_x = img_w // 2
            center_y = img_h // 2

        x1 = center_x - (crop_w // 2)
        y1 = center_y - (crop_h // 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        pad_left = max(0, -x1); pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w); pad_bottom = max(0, y2 - img_h)
        
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            new_w, new_h = img_w + pad_left + pad_right, img_h + pad_top + pad_bottom
            expanded_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))
            expanded_img.paste(pil_img, (pad_left, pad_top))
            pil_img = expanded_img
            x1 += pad_left; y1 += pad_top; x2 += pad_left; y2 += pad_top
        else:
            if x1 < 0: x1 = 0; x2 = crop_w
            if y1 < 0: y1 = 0; y2 = crop_h
            if x2 > pil_img.width: x2 = pil_img.width; x1 = max(0, x2 - crop_w)
            if y2 > pil_img.height: y2 = pil_img.height; y1 = max(0, y2 - crop_h)
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped = pil_img.crop((x1, y1, x2, y2))
        final_img = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        debug_img = None
        if debug_mode:
            debug_img = pil_img.copy()
            draw = ImageDraw.Draw(debug_img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            if detected_face is not None:
                fx, fy, fw, fh = detected_face
                if pad_left > 0 or pad_top > 0: fx+=pad_left; fy+=pad_top
                draw.rectangle([fx, fy, fx+fw, fy+fh], outline="#00ff00", width=3)
        
        return final_img, debug_img

# =============================================================================
# 8. SHEET GENERATOR
# =============================================================================
def generate_passport_sheet(photo_img, paper_size="A4", orientation="Portrait", spacing=10):
    PAPER_SIZES = { "A4": (2480, 3508), "4x6 inch": (1200, 1800) }
    sheet_w, sheet_h = PAPER_SIZES.get(paper_size, PAPER_SIZES["A4"])
    if orientation == "Landscape": sheet_w, sheet_h = sheet_h, sheet_w
    
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    p_w, p_h = photo_img.size
    
    cols = (sheet_w - spacing) // (p_w + spacing)
    rows = (sheet_h - spacing) // (p_h + spacing)
    start_x = (sheet_w - (cols * p_w + (cols - 1) * spacing)) // 2
    start_y = (sheet_h - (rows * p_h + (rows - 1) * spacing)) // 2
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            x = int(start_x + c * (p_w + spacing))
            y = int(start_y + r * (p_h + spacing))
            sheet.paste(photo_img, (x, y))
            draw.rectangle([x-1, y-1, x+p_w+1, y+p_h+1], outline="#DDDDDD", width=1)
            count += 1
    return sheet, count

# =============================================================================
# 9. BULK PROCESSING
# =============================================================================
def process_bulk(files, settings, ai_engine):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(files)
        
        for i, file in enumerate(files):
            try:
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / total)
                
                file.seek(0)
                file_bytes = file.read()
                
                img = get_processed_image(
                    file_bytes, 
                    settings['brightness'], settings['contrast'], settings['sharpness'],
                    settings['bg_mode'], settings['bg_custom']
                )
                
                if img is None: continue
                if settings['stabilize']: img = cv2_to_pil(CropEngine.stabilize_image(pil_to_cv2(img)))
                
                processed_img, _ = CropEngine.smart_crop(
                    img, settings['width'], settings['height'], ai_engine, settings['use_ai'], False
                )
                
                img_byte_arr = io.BytesIO()
                fmt = settings['format'].upper()
                if fmt == 'JPG': fmt = 'JPEG'
                if fmt == 'PDF': 
                    processed_img.save(img_byte_arr, format='PDF', resolution=100.0)
                elif fmt in ['JPEG', 'WEBP', 'TIFF']:
                    processed_img.save(img_byte_arr, format=fmt, quality=95)
                else: 
                    processed_img.save(img_byte_arr, format=fmt)
                
                fname = f"bigsammy_crop_{i+1:03d}_{file.name.rsplit('.', 1)[0]}.{settings['format'].lower()}"
                zip_file.writestr(fname, img_byte_arr.getvalue())
                
                # Explicit Garbage Collection
                del img, processed_img
                gc.collect()
                
            except Exception as e:
                st.error(f"Error: {e}")
        status_text.text("Done!")
    return zip_buffer.getvalue()

def get_image_download_link(img, format_str):
    buf = io.BytesIO()
    fmt = format_str.upper()
    if fmt == "JPG": fmt = "JPEG"
    if fmt in ["JPEG", "WEBP"]: img.save(buf, format=fmt, quality=95)
    else: img.save(buf, format=fmt)
    return buf.getvalue()

# =============================================================================
# 10. USER INTERFACE
# =============================================================================
def main():
    apply_ghanaian_theme()
    hide_streamlit_elements() # STEALTH MODE ACTIVATED
    
    ai_engine = IntelligenceEngine()
    
    st.markdown('<div class="decoration-line"></div>', unsafe_allow_html=True)
    st.title("BigSammyCrop | Smart African Precision")
    st.markdown("**Akwaaba (Welcome).** Upload your images for intelligent, precise resizing.")
    
    if 'confirmed_crop_img' not in st.session_state: st.session_state.confirmed_crop_img = None
    if 'last_settings_hash' not in st.session_state: st.session_state.last_settings_hash = ""
    # Initialize uploader key for flushing mechanism
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

    # -- Sidebar --
    st.sidebar.header("⚙️ Configuration")
    
    # NEW: RESET BUTTON (Clears RAM & Browser Cache)
    if st.sidebar.button("🧹 Start Over (Clear Memory)", type="primary"):
        st.session_state.confirmed_crop_img = None
        st.session_state.last_settings_hash = ""
        st.session_state.uploader_key += 1 # Forces browser to drop files
        gc.collect()
        st.rerun()
    
    # Dynamic File Uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Images", 
        type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    # 1. CROP
    st.sidebar.subheader("📐 Crop Dimensions")
    preset_choice = st.sidebar.selectbox("Choose Preset", list(PRESETS.keys()), index=2)
    if preset_choice == "Custom Dimensions":
        c1, c2 = st.sidebar.columns(2)
        target_w = c1.number_input("Width (px)", 10, 8000, 800)
        target_h = c2.number_input("Height (px)", 10, 8000, 600)
    else:
        target_w, target_h = PRESETS[preset_choice]
        st.sidebar.info(f"Target: {target_w} x {target_h}")

    # 2. BACKGROUND LAB
    with st.sidebar.expander("✂️ Background Lab", expanded=True):
        if HAS_REMBG:
            bg_mode = st.radio("Background Style", ["Original", "White", "Blue", "Red", "Transparent", "Custom"])
            bg_custom = "#FFFFFF"
            if bg_mode == "Custom":
                bg_custom = st.color_picker("Pick Color", "#FFFFFF")
        else:
            st.warning("⚠️ Install 'rembg' to use this feature.")
            bg_mode = "Original"
            bg_custom = "#FFFFFF"

    # 3. STUDIO LAB
    with st.sidebar.expander("🎨 Studio Lab (Enhance)", expanded=False):
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)

    # 4. INTELLIGENCE & MANUAL
    st.sidebar.subheader("🤖 Intelligence")
    use_ai = st.sidebar.checkbox("AI Subject Centering", value=True)
    stabilize = st.sidebar.checkbox("Auto-Straighten", value=False)
    
    manual_crop_mode = False
    if uploaded_files and len(uploaded_files) == 1 and HAS_CROPPER:
        manual_crop_mode = st.sidebar.checkbox("🛠️ Manual Crop Mode", value=False)

    debug_mode = st.sidebar.checkbox("Show AI Vision", value=False)
    # TIFF Added
    export_format = st.sidebar.selectbox("Export Format", ["JPG", "PNG", "WEBP", "TIFF", "PDF"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("👨‍💻 Developer Contact")
    st.sidebar.markdown("**Samuel Kwame Dordzie**")
    st.sidebar.markdown("📧 bigsammy86g@gmail.com")
    st.sidebar.markdown("📱 WhatsApp: 0246652528")
    st.sidebar.caption("© 2026 BigSammy Graphics Consult")

    if not uploaded_files:
        st.info("👋 Upload an image to begin.")
        return

    settings = {
        'width': target_w, 'height': target_h, 'use_ai': use_ai, 
        'stabilize': stabilize, 'format': export_format,
        'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness,
        'bg_mode': bg_mode, 'bg_custom': bg_custom
    }
    
    current_hash = f"{target_w}-{target_h}-{bg_mode}-{bg_custom}-{brightness}-{contrast}-{uploaded_files[0].name}"
    if st.session_state.last_settings_hash != current_hash:
        st.session_state.confirmed_crop_img = None
        st.session_state.last_settings_hash = current_hash

    # --- SINGLE IMAGE MODE ---
    if len(uploaded_files) == 1:
        f = uploaded_files[0]
        f.seek(0)
        file_bytes = f.read()
        
        with st.spinner("Processing Background & Enhancements..."):
            processed_source = get_processed_image(
                file_bytes, brightness, contrast, sharpness, bg_mode, bg_custom
            )
        
        if processed_source is None:
            st.error("Error loading image."); return

        col1, col2 = st.columns(2)
        
        # LEFT: SOURCE
        with col1:
            st.markdown("### Source")
            if manual_crop_mode and HAS_CROPPER:
                st.info("Adjust box. Click Apply.")
                cropped_preview = st_cropper(
                    processed_source, realtime_update=True, box_color='#CE1126',
                    aspect_ratio=(target_w, target_h), should_resize_image=True,
                    key="manual_cropper"
                )
                if st.button("✅ Apply Crop & Process", width="stretch"):
                    final_crop = cropped_preview.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    st.session_state.confirmed_crop_img = final_crop
            else:
                if not st.session_state.confirmed_crop_img:
                    with st.spinner("Calculating AI Crop..."):
                        img_to_process = processed_source
                        if stabilize: 
                            img_to_process = cv2_to_pil(CropEngine.stabilize_image(pil_to_cv2(processed_source)))
                        
                        processed_img, debug_img = CropEngine.smart_crop(
                            img_to_process, target_w, target_h, ai_engine, use_ai, debug_mode
                        )
                        st.session_state.confirmed_crop_img = processed_img
                        if debug_mode and debug_img: st.image(debug_img, width="stretch")
                        else: st.image(processed_source, width="stretch")
                else:
                    st.image(processed_source, width="stretch")

        # RIGHT: RESULT
        with col2:
            st.markdown("### Processed Result")
            final_image = st.session_state.confirmed_crop_img
            
            if final_image:
                st.image(final_image, width="stretch")
                img_bytes = get_image_download_link(final_image, export_format)
                st.download_button(f"⬇️ Download Image", img_bytes, f"bigsammy_crop.{export_format.lower()}", f"image/{export_format.lower()}", width="stretch")
                
                st.markdown("---")
                st.subheader("🖨️ Print Sheet")
                c1, c2, c3 = st.columns([1,1,1])
                p_size = c1.selectbox("Size", ["A4", "4x6 inch"])
                orient = c2.selectbox("Orientation", ["Portrait", "Landscape"])
                space = c3.number_input("Space", 5, 50, 10)
                
                if st.button("Generate Sheet", width="stretch"):
                    sheet, count = generate_passport_sheet(final_image, p_size, orient, space)
                    st.success(f"{count} photos on {p_size}")
                    st.image(sheet, width="stretch")
                    sheet_bytes = get_image_download_link(sheet, "JPG")
                    st.download_button("⬇️ Download Sheet", sheet_bytes, "print_sheet.jpg", "image/jpeg", width="stretch")
            else:
                st.info("Waiting for Apply...")

    # --- BULK MODE ---
    else:
        st.subheader(f"📦 Bulk Processing: {len(uploaded_files)} Images")
        if st.button("🚀 Start Bulk Process", width="stretch", type="primary"):
            with st.spinner("Processing (this may take time with BG removal)..."):
                zip_bytes = process_bulk(uploaded_files, settings, ai_engine)
                st.download_button("⬇️ Download ZIP", zip_bytes, "bulk_crop.zip", "application/zip", width="stretch")
        
        # Explicit GC after bulk runs
        gc.collect()

if __name__ == "__main__":
    main()
