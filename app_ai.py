import streamlit as st
import numpy as np
import hashlib
import cv2  # OpenCV for DCT
import io
from PIL import Image

# ==========================================
# 1. THE AI DECISION LAYER
# ==========================================
class AISmartSelector:
    """
    Analyzes the image structure to decide the best hiding strategy.
    """
    def analyze_and_select(self, image_file):
        # Read file info
        file_bytes = image_file.getvalue()
        
        # 1. Check Magic Numbers (File Headers)
        # PNG starts with: 89 50 4E 47
        # JPG starts with: FF D8 FF
        hex_header = file_bytes[:4].hex().upper()
        
        strategy = "UNKNOWN"
        reason = ""

        if hex_header.startswith("89504E47"):
            strategy = "LSB"
            reason = "Lossless Format (PNG) detected. Optimized for High Capacity."
        elif hex_header.startswith("FFD8"):
            strategy = "DCT"
            reason = "Compressed Format (JPG) detected. Switching to Robust DCT Mode."
        else:
            # Fallback for BMP/TIFF or other formats
            strategy = "LSB" 
            reason = "Unknown format (likely Raw/BMP). Defaulting to LSB."

        return strategy, reason

# ==========================================
# 2. STEGANOGRAPHY ENGINES
# ==========================================

class LSBEngine:
    """Standard LSB for PNGs"""
    def embed(self, image, data):
        image = image.convert("RGB")
        encoded = image.copy()
        width, height = image.size
        
        full_msg = data + "#####"
        bits = ''.join(format(ord(i), '08b') for i in full_msg)
        
        if len(bits) > width * height * 3:
            raise ValueError("Message too large for LSB.")
            
        idx = 0
        pixels = encoded.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if idx < len(bits):
                    r = r & ~1 | int(bits[idx])
                    idx += 1
                if idx < len(bits):
                    g = g & ~1 | int(bits[idx])
                    idx += 1
                if idx < len(bits):
                    b = b & ~1 | int(bits[idx])
                    idx += 1
                pixels[x, y] = (r, g, b)
                if idx >= len(bits): break
            if idx >= len(bits): break
        return encoded

    def extract(self, image):
        bits = ""
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                r, g, b = pixels[x, y]
                bits += str(r & 1)
                bits += str(g & 1)
                bits += str(b & 1)
        
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        res = ""
        for c in chars:
            try:
                res += chr(int(c, 2))
                if res.endswith("#####"): return res[:-5]
            except: break
        return ""

class DCTEngine:
    """
    Robust Steganography for JPGs using Discrete Cosine Transform.
    Hides data in the frequency coefficients (middle frequencies).
    """
    def embed(self, image_file, data):
        # Convert to cv2 format
        file_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        # Resize to be 8x8 compatible
        h, w = img.shape[:2]
        h = (h // 8) * 8
        w = (w // 8) * 8
        img = img[:h, :w]
        
        # Work on Blue channel (least noticeable)
        b, g, r = cv2.split(img)
        img_float = np.float32(b)
        
        # Prepare Message
        full_msg = data + "#####"
        bits = ''.join(format(ord(i), '08b') for i in full_msg)
        bit_idx = 0
        
        # Block Processing
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                if bit_idx >= len(bits): break
                
                # Get 8x8 block
                block = img_float[i:i+8, j:j+8]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Hide bit in (4,4) coefficient (Mid-frequency)
                # Logic: If bit is 1, make coeff positive. If 0, make negative.
                # (This is a simplified robust strategy)
                
                val = dct_block[4, 4]
                if bits[bit_idx] == '1':
                    if val < 0: val = -val
                    if val < 5: val = 10 # Reinforce
                else: # bit is 0
                    if val > 0: val = -val
                    if val > -5: val = -10 # Reinforce
                
                dct_block[4, 4] = val
                
                # Inverse DCT
                img_float[i:i+8, j:j+8] = cv2.idct(dct_block)
                bit_idx += 1
                
        # Merge Channels
        img_out = cv2.merge((np.uint8(img_float), g, r))
        return img_out

    def extract(self, image_file):
        # Convert to cv2
        file_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        h, w = img.shape[:2]
        h = (h // 8) * 8
        w = (w // 8) * 8
        img = img[:h, :w]
        
        b, g, r = cv2.split(img)
        img_float = np.float32(b)
        
        bits = ""
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = img_float[i:i+8, j:j+8]
                dct_block = cv2.dct(block)
                
                val = dct_block[4, 4]
                if val > 0: bits += "1"
                else: bits += "0"
        
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        res = ""
        for c in chars:
            try:
                res += chr(int(c, 2))
                if res.endswith("#####"): return res[:-5]
            except: pass
        return res

# ==========================================
# 3. CORE ENCRYPTION (Unchanged)
# ==========================================
class KyberEngine:
    def encrypt(self, pub, msg):
        # Simulated encryption
        return "ENC_" + msg[::-1] 
    def decrypt(self, priv, cipher):
        return cipher[4:][::-1]

# ==========================================
# 4. MAIN APP UI
# ==========================================
st.set_page_config(page_title="QuantumGuard AI", layout="wide")
st.title("🧠 QuantumGuard: AI-Adaptive Mode")

# Initialize
ai = AISmartSelector()
lsb = LSBEngine()
dct = DCTEngine()
kyber = KyberEngine()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Sender (AI-Driven)")
    msg = st.text_input("Message", "Secret Launch Codes")
    uploaded_file = st.file_uploader("Upload ANY Image (JPG or PNG)", type=["png", "jpg", "jpeg"])
    
    if st.button("Analyze & Send"):
        if uploaded_file and msg:
            # Step A: AI Analysis
            strategy, reason = ai.analyze_and_select(uploaded_file)
            
            st.info(f"🤖 **AI Decision:** {reason}")
            
            # Step B: Encryption
            cipher = kyber.encrypt("pubkey", msg)
            
            # Step C: Adaptive Steganography
            out_buffer = io.BytesIO()
            
            if strategy == "LSB":
                img = Image.open(uploaded_file)
                stego_img = lsb.embed(img, cipher)
                stego_img.save(out_buffer, format="PNG")
                st.success(f"✅ Embedded using LSB Protocol.")
                
            elif strategy == "DCT":
                # Reset file pointer for OpenCV reading
                uploaded_file.seek(0)
                stego_cv2 = dct.embed(uploaded_file, cipher)
                # Encode back to stream
                is_success, buffer = cv2.imencode(".png", stego_cv2) # Save as PNG to prevent double compression loss
                out_buffer = io.BytesIO(buffer)
                st.success(f"✅ Embedded using DCT Protocol.")
            
            st.download_button("Download Secure Image", out_buffer.getvalue(), "secure_output.png")

with col2:
    st.header("2. Receiver")
    rx_file = st.file_uploader("Upload Received Image", type=["png", "jpg"])
    
    if st.button("Decrypt"):
        if rx_file:
            # AI detects method automatically (based on success?) 
            # Or we can try both. For this demo, we assume PNG output from Sender.
            # But let's verify if AI detects it.
            
            try:
                # Try LSB first
                img = Image.open(rx_file)
                extracted = lsb.extract(img)
                
                if "ENC_" not in extracted:
                    # If LSB fails, Try DCT
                    rx_file.seek(0)
                    extracted = dct.extract(rx_file)
                
                if "ENC_" in extracted:
                    plain = kyber.decrypt("priv", extracted)
                    st.success(f"🔓 Decrypted: {plain}")
                else:
                    st.error("Failed to recover message.")
            except Exception as e:
                st.error(f"Error: {e}")