import io
import json
import re
from typing import List, Optional, Tuple, Dict, Any

# Core libraries for HTTP, image processing, and PDF conversion
import requests
import numpy as np
import cv2
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode
#from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup

# FastAPI specific imports
from fastapi import FastAPI, Form, UploadFile, HTTPException, status
from fastapi.responses import PlainTextResponse

# --- CONFIGURATION ---
USER_AGENT = "HackathonVerifier/1.0"
REQUEST_TIMEOUT = 8  # seconds
PDF_DPI = 300  # resolution when converting PDF page to image


# -------------------------
# CORE HELPER FUNCTIONS (The Extraction Engine)
# -------------------------

# Helper 1: Load file bytes into PIL Image for processing
def load_image_from_bytes(file_bytes: bytes, filename: str) -> Image.Image:
    """Return a PIL RGB image from uploaded image or pdf (first page)."""
    file_stream = io.BytesIO(file_bytes)
    filename = filename.lower()

    # Handle PDF (Requires Poppler)
    '''if filename.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        try:
            pages = convert_from_bytes(file_bytes, dpi=PDF_DPI, first_page=1, last_page=1)
            if not pages:
                raise ValueError("No pages found in PDF.")
            return pages[0].convert("RGB")
        except Exception as e:
            # This handles the case where Poppler is missing
            raise RuntimeError(f"PDF conversion failed (Is Poppler installed?): {e}")
    '''
    # Handle Image
    try:
        img = Image.open(file_stream).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Image loading failed: {e}")


# Helper 2: Extract QR payload strings from PIL image
def extract_qrcodes_from_image(pil_img: Image.Image) -> List[str]:
    """Extract QR payloads from PIL image using pyzbar and OpenCV fallback."""
    results: List[str] = []
    
    # Try pyzbar
    try:
        decoded = pyzbar_decode(pil_img)
        for d in decoded:
            if d and d.data:
                results.append(d.data.decode("utf-8", errors="ignore"))
    except Exception:
        pass

    # OpenCV fallback
    if not results:
        try:
            arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            qdet = cv2.QRCodeDetector()
            sdata, _, _ = qdet.detectAndDecode(arr)
            if sdata:
                results.append(sdata)
        except Exception:
            pass
            
    out: List[str] = []
    seen = set()
    for r in results:
        if r not in seen:
            out.append(r.strip())
            seen.add(r)
    return out


# Helper 3: Fetch URL and aggressively parse for JSON/Name
def fetch_json_from_url(url: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Try to GET the URL and return (parsed_json_or_none, info_dict)."""
    info: Dict[str, Any] = {"url": url}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        # 1. Try direct JSON parse
        try:
            j = resp.json()
            if isinstance(j, (dict, list)):
                return j, info
        except Exception:
            pass

        # 2. Try to find JSON embedded in <script type="application/ld+json"> (Schema Markup)
        if "html" in resp.headers.get("Content-Type", "").lower() or "<html" in (resp.text or "").lower():
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup.find_all("script", {"type": "application/ld+json"}):
                try:
                    j = json.loads(script.string or "")
                    return j, info
                except Exception:
                    continue

    except Exception as e:
        info["error"] = str(e)

    return None, info


# Helper 4: Deep search for a recipient name in JSON data
NAME_KEYS = {
    "name", "full_name", "fullname", "fullName", "candidateName", "candidate_name",
    "studentName", "student_name", "recipient", "issuedTo", "displayName", "personName"
}

def extract_name_from_json(obj: Any) -> Optional[str]:
    """Deep search a JSON-like object for likely name fields."""
    found_names: List[str] = []

    def deep_search(o: Any):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in NAME_KEYS and isinstance(v, str) and v.strip():
                    found_names.append(v.strip())
                else:
                    deep_search(v)
        elif isinstance(o, list):
            for item in o:
                deep_search(item)

    deep_search(obj)
    
    if found_names:
        found_names.sort(key=len, reverse=True)
        return found_names[0]
    
    return None


# -------------------------
# FASTAPI APPLICATION AND ENDPOINT (Scan & Verify)
# -------------------------

app = FastAPI(
    title="Certificate Verification API (Full Scan)",
    description="Scans uploaded certificate image/PDF, extracts data from QR, and verifies inputs.",
)

@app.post("/verify-certificate", 
          status_code=status.HTTP_200_OK,
          response_class=PlainTextResponse,
          summary="Performs QR scan, extracts name, and returns final VERIFIED status.")
async def verify_certificate_endpoint(
    # Input 1: The certificate file itself (for scanning)
    file: UploadFile = Form(..., description="Certificate image (PNG, JPG, or PDF)"),
    
    # Input 2: The name typed by the end-user (to be checked against the QR name)
    recipient_name_input: str = Form(..., description="Recipient Name provided by the user (for verification)"),
    
    # Input 3: The visible link typed by the end-user (to be checked against the QR link)
    visible_link_input: str = Form(..., description="The visible link written on the certificate (e.g., https://postman.com)")
):
    
    file_bytes = await file.read()
    
    # Initialize variables
    extracted_name: Optional[str] = None
    qr_payload: Optional[str] = None
    
    try:
        # 1. Load File and Image Preparation (Scanning starts here)
        pil_img = load_image_from_bytes(file_bytes, file.filename)
        
        # 2. Extract QR Code Payload
        qr_payloads = extract_qrcodes_from_image(pil_img)
        if not qr_payloads:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                                detail="No QR codes were detected in the provided file.")

        qr_payload = qr_payloads[0].strip()
        
        # 3. Process Payload (Fetch Verification Data)
        parsed_json = None
        
        # Aggressively check payload as URL, direct JSON, or embedded URL
        if re.match(r"^https?://", qr_payload, flags=re.I):
            parsed_json, _ = fetch_json_from_url(qr_payload)
        
        else:
            # Fallback for direct JSON in payload
            try:
                parsed_json = json.loads(qr_payload.strip().strip('"'))
            except Exception:
                # Search for URL inside the payload text
                m = re.search(r"(https?://[^\s'\"]+)", qr_payload)
                if m:
                    url_inside = m.group(1)
                    parsed_json, _ = fetch_json_from_url(url_inside)
        
        # 4. Extract Name from Fetched/Parsed Data
        if parsed_json:
            extracted_name = extract_name_from_json(parsed_json)
        
    except HTTPException:
        # Re-raise explicit HTTP exceptions
        raise
    except Exception as e:
        # Catch internal processing errors
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
                            detail=f"Processing error: {type(e).__name__} - {str(e)}")

    
    # --- 5. Verification Logic (Comparison) ---
    
    is_name_match = False
    is_link_match = False
    
    # Check 1: Name Match (The core comparison)
    if extracted_name and recipient_name_input:
        normalized_extracted = re.sub(r'[^a-z0-9]', '', extracted_name.lower())
        normalized_input = re.sub(r'[^a-z0-9]', '', recipient_name_input.lower())
        
        if normalized_extracted == normalized_input or normalized_input in normalized_extracted:
            is_name_match = True

    # Check 2: Link Match 
    if qr_payload and visible_link_input:
        if visible_link_input.lower() in qr_payload.lower():
             is_link_match = True

    # --- 6. Final Status ---
    final_verification_status = "VERIFIED" if is_name_match and is_link_match else "NOT VERIFIED"

    # Return the string directly as requested
    return final_verification_status