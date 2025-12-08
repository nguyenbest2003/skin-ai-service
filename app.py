# app.py - Skin Analyzer PRO+ v5.0
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import numpy as np
import torchvision.transforms as T
import torch, cv2, io, os, json, datetime, uuid, time
from torch.nn import functional as F
import logging
from functools import lru_cache

# optional Mediapipe for robust face mesh
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    mp_face = mp.solutions.face_mesh
except Exception:
    HAS_MEDIAPIPE = False

# optional Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if genai and GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

logging.basicConfig(level=logging.INFO, format="[SkinAI] %(message)s")
log = logging.getLogger("SkinAI")

app = FastAPI(title="Skin Analyzer PRO+ v5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

BASE_TMP = Path("tmp"); BASE_TMP.mkdir(exist_ok=True, parents=True)

def now_iso(): return datetime.datetime.utcnow().isoformat() + "Z"
def make_analysis_dir(aid: str) -> Path:
    p = BASE_TMP / aid; p.mkdir(parents=True, exist_ok=True); return p

# ---------- lightweight model wrapper (kept for GradCAM layers) ----------
class SkinAnalyzerModel:
    def __init__(self):
        log.info("Loading ResNet50 (for explainability only)...")
        try:
            self.model = torch.hub.load("pytorch/vision:v0.15.2", "resnet50", pretrained=True)
            self.model.eval()
        except Exception as e:
            log.warning("Could not load resnet (explainability disabled): %s", e)
            self.model = None
        self.transform = T.Compose([T.Resize((224,224)), T.ToTensor(),
                                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        # hooks
        self.gradients = None; self.activations = None
        if self.model:
            try:
                layer = self.model.layer4[-1].conv3
                def save_activation(module, inp, out): self.activations = out
                def save_gradient(module, grad_in, grad_out): self.gradients = grad_out[0]
                layer.register_forward_hook(save_activation)
                try:
                    layer.register_full_backward_hook(save_gradient)
                except Exception:
                    layer.register_backward_hook(save_gradient)
            except Exception as e:
                log.warning("GradCAM hooks not registered: %s", e)

    def preprocess(self, data: bytes):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        t = self.transform(img).unsqueeze(0)
        return t, img

    def predict_score(self, tensor):
        if self.model is None:
            return 0.0
        with torch.no_grad():
            out = self.model(tensor)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        return float(probs.max())

    def gradcam_overlay(self, tensor, pil_img, outpath):
        if self.model is None:
            pil_img.save(outpath)
            return outpath
        self.model.zero_grad()
        out = self.model(tensor)
        score = out[0].max()
        score.backward()
        if self.gradients is None or self.activations is None:
            pil_img.save(outpath); return outpath
        grad = self.gradients.detach().cpu().numpy()
        act = self.activations.detach().cpu().numpy()
        if grad.ndim==4: grad=grad[0]
        if act.ndim==4: act=act[0]
        weights = grad.mean(axis=(1,2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for w,a in zip(weights,act): cam += w*a
        cam = np.maximum(cam,0)
        cam = (cam - cam.min())/(cam.max()+1e-8)
        camr = cv2.resize(cam, (pil_img.width, pil_img.height))
        heat = np.uint8(camr*255)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr,0.6,heat,0.4,0)
        cv2.imwrite(outpath, overlay)
        return outpath

model_wrapper = SkinAnalyzerModel()

# ---------- Face zones utils ----------
# If mediapipe available, use mesh landmarks to define polygons; else fallback to bbox-based zones
def get_face_landmarks(pil_img):
    if not HAS_MEDIAPIPE:
        return None
    img = np.array(pil_img)[:,:,::-1]  # RGB->BGR
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(img)
    if not res or not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    h,w = pil_img.height, pil_img.width
    pts = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
    return pts

# Index groups for rough zones if using mediapipe (common mapping)
# We'll use approximate groups for forehead, nose, left_cheek, right_cheek, chin
MP_ZONE_IDX = {
    "forehead": list(range(10, 20)),      # approximate
    "nose": list(range(1, 6)),
    "left_cheek": list(range(205, 220)),
    "right_cheek": list(range(425, 440)),
    "chin": list(range(152, 168))
}

def polygon_from_indices(pts, idxs):
    poly = [pts[i] for i in idxs if i < len(pts)]
    return poly if len(poly)>=3 else None

# Fallback bbox-based zones (x,y,w,h) in % of image
FALLBACK_ZONES = {
    "forehead": (0.2, 0.03, 0.6, 0.18),
    "nose": (0.38, 0.25, 0.24, 0.25),
    "left_cheek": (0.05, 0.25, 0.35, 0.4),
    "right_cheek": (0.6, 0.25, 0.35, 0.4),
    "chin": (0.3, 0.65, 0.4, 0.3)
}

def mask_from_polygon(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygon:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

def mask_from_bbox(shape, bbox_percent):
    h,w = shape[:2]
    x,y,ww,hh = bbox_percent
    x1 = int(x*w); y1 = int(y*h); x2 = int((x+ww)*w); y2 = int((y+hh)*h)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
    return mask

# ---------- CV metrics per region ----------
def acne_detection_region(np_bgr, region_mask):
    # use YCrCb Cr channel on masked region
    ycrcb = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:,:,1]
    cr_masked = cv2.bitwise_and(cr, cr, mask=region_mask)
    blur = cv2.GaussianBlur(cr_masked, (5,5), 0)
    # adapt threshold on local region only
    # normalize small area to 0..255 then threshold
    if region_mask.sum()==0:
        return 0.0, 0
    blur_norm = cv2.normalize(blur, None, 0,255,cv2.NORM_MINMAX)
    thresh = cv2.adaptiveThreshold(np.uint8(blur_norm),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,3)
    # apply mask again
    thresh = cv2.bitwise_and(thresh, thresh, mask=region_mask)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = [c for c in cnts if 8 < cv2.contourArea(c) < 1000]
    score = min(1.0, len(spots) / 50)
    return float(score), len(spots)

def redness_region(np_bgr, region_mask):
    lab = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:,:,1]
    a_masked = cv2.bitwise_and(a,a,mask=region_mask)
    if region_mask.sum()==0: return 0.0
    val = (a_masked[a_masked>0].mean() - 128) / 50 if (a_masked>0).any() else 0.0
    return float(max(0, min(1, val)))

def oiliness_region(np_bgr, region_mask):
    hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    v_masked = cv2.bitwise_and(v,v,mask=region_mask)
    if region_mask.sum()==0: return 0.0
    bright_pct = float((v_masked>200).sum()) / (region_mask.sum()/255)
    return float(min(1.0, bright_pct*3))

def pore_region(np_bgr, region_mask):
    gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=region_mask)
    if region_mask.sum()==0: return 0.0
    blur1 = cv2.GaussianBlur(gray_masked, (3,3),0)
    blur2 = cv2.GaussianBlur(gray_masked, (11,11),0)
    dog = cv2.absdiff(blur1, blur2)
    score = float(min(1.0, np.mean(dog)/20))
    return score

def melanin_region(np_bgr, region_mask):
    # detect darker spots: low V and somewhat high saturation
    hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    v_masked = cv2.bitwise_and(v,v,mask=region_mask)
    s_masked = cv2.bitwise_and(s,s,mask=region_mask)
    # threshold: v low and s moderate
    mask_dark = np.zeros_like(v_masked)
    mask_dark[(v_masked<80) & (s_masked>30)] = 255
    cnts, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = [c for c in cnts if 5 < cv2.contourArea(c) < 1000]
    score = float(min(1.0, len(spots)/40))
    return score, len(spots)

# ---------- Layer generation ----------
def generate_layer_image(np_bgr, regions_info):
    # create empty layers (H,W,3)
    h,w = np_bgr.shape[:2]
    layer_acne = np.zeros((h,w,3), dtype=np.uint8)
    layer_red = np.zeros_like(layer_acne)
    layer_oil = np.zeros_like(layer_acne)
    # draw per-region heat intensities
    for zone, info in regions_info.items():
        mask = info['mask']
        # intensity in 0..1
        acne_i = int(info['acne_score']*255)
        red_i = int(info['redness']*255)
        oil_i = int(info['oiliness']*255)
        # acne -> red channel
        layer_acne[:,:,2] = np.where(mask==255, np.maximum(layer_acne[:,:,2], acne_i), layer_acne[:,:,2])
        # red -> green channel
        layer_red[:,:,1] = np.where(mask==255, np.maximum(layer_red[:,:,1], red_i), layer_red[:,:,1])
        # oil -> blue channel
        layer_oil[:,:,0] = np.where(mask==255, np.maximum(layer_oil[:,:,0], oil_i), layer_oil[:,:,0])
    # colorize slightly and overlay to original for combined visuals done elsewhere
    return layer_acne, layer_red, layer_oil
# ---------- Summary & Skin Type ----------
def summarize_skin_vi(acne, red, oil, pore):
    parts = []

    # Mụn
    if acne > 0.6:
        parts.append("Da có nhiều mụn viêm và mụn đầu đen.")
    elif acne > 0.3:
        parts.append("Da có mụn mức độ trung bình.")
    else:
        parts.append("Da ít mụn, khá ổn định.")

    # Độ đỏ
    if red > 0.3:
        parts.append("Da dễ đỏ và có dấu hiệu nhạy cảm.")
    else:
        parts.append("Độ đỏ thấp, ít kích ứng.")

    # Dầu
    if oil > 0.5:
        parts.append("Da dầu nhiều, đặc biệt vùng chữ T.")
    elif oil > 0.3:
        parts.append("Da hơi dầu.")
    else:
        parts.append("Da khô – bình thường.")

    # Lỗ chân lông
    if pore > 0.4:
        parts.append("Lỗ chân lông to và dễ thấy.")
    else:
        parts.append("Lỗ chân lông nhỏ – trung bình.")

    return " ".join(parts)


def classify_skin_type(acne, red, oil, pore):
    if red > 0.35:
        return "Da nhạy cảm"
    if oil > 0.45 and pore > 0.35:
        return "Da dầu – lỗ chân lông to"
    if oil > 0.35:
        return "Da dầu"
    if oil < 0.2 and acne < 0.3:
        return "Da khô – bình thường"
    return "Da hỗn hợp"

# ---------- LLM helper (reuse previous gemini_call_cached) ----------
CACHE_SIZE = 2048
@lru_cache(maxsize=CACHE_SIZE)
def gemini_call_cached(acne, red, oil, pore, per_zone=False):
    prompt = f"Bạn là chuyên gia da liễu. Tổng quan: acne={acne}, red={red}, oil={oil}, pore={pore}.\n"
    if per_zone:
        prompt += "Trả lời theo từng vùng: FOREHEAD, NOSE, LEFT_CHEEK, RIGHT_CHEEK, CHIN.\n"
    prompt += "Ngắn gọn, hành động thực tế (sáng/tối/thành phần), tiếng Việt."
    if genai is None:
        return "Gemini không được cấu hình. Vui lòng cấu hình GEMINI_API_KEY hoặc dùng offline fallback."
    primary = "models/gemini-2.0-flash-lite"
    fallback = "models/gemma-3-4b-it"
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(primary)
            res = model.generate_content(prompt)
            text = getattr(res, "text", None) or (res.get("candidates")[0].get("content") if isinstance(res, dict) else None)
            if text:
                return text
        except Exception as e:
            log.warning("Gemini err: %s", e); time.sleep(0.8*(2**attempt))
    try:
        model2 = genai.GenerativeModel(fallback)
        res2 = model2.generate_content(prompt)
        text2 = getattr(res2,"text",None) or (res2.get("candidates")[0].get("content") if isinstance(res2, dict) else None)
        if text2: return text2
    except Exception as e:
        log.error("Gemini fallback failed: %s", e)
    # local fallback
    return "AI tạm thời không khả dụng — đây là gợi ý sơ bộ: giữ vệ sinh, tránh nặn, dùng sản phẩm dịu nhẹ."

# ---------- MAIN ANALYZER ----------
analyzer = model_wrapper

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    """Full analysis v5.1: NO heatmap, only metrics + summary."""
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, "File empty")

        aid = uuid.uuid4().hex
        adir = make_analysis_dir(aid)

        # save original image
        orig_path = adir / "image.jpg"
        with open(orig_path, "wb") as f:
            f.write(data)

        # preprocess
        tensor, pil_img = analyzer.preprocess(data)
        np_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # face landmarks
        pts = get_face_landmarks(pil_img) if HAS_MEDIAPIPE else None

        # per-zone analysis
        regions_info = {}
        for zone in ["forehead", "nose", "left_cheek", "right_cheek", "chin"]:
            if pts:
                poly = polygon_from_indices(pts, MP_ZONE_IDX.get(zone, []))
                mask = mask_from_polygon(np_bgr.shape, poly) if poly else mask_from_bbox(np_bgr.shape, FALLBACK_ZONES[zone])
            else:
                mask = mask_from_bbox(np_bgr.shape, FALLBACK_ZONES[zone])

            acne_s, acne_count = acne_detection_region(np_bgr, mask)
            red_s = redness_region(np_bgr, mask)
            oil_s = oiliness_region(np_bgr, mask)
            pore_s = pore_region(np_bgr, mask)
            mel_s, mel_count = melanin_region(np_bgr, mask)

            regions_info[zone] = {
                "mask": None,  # remove mask from result
                "acne_score": acne_s,
                "acne_count": acne_count,
                "redness": red_s,
                "oiliness": oil_s,
                "pore_score": pore_s,
                "melanin_score": mel_s,
                "melanin_spots": mel_count
            }

        # GLOBAL
        acne_global = np.mean([regions_info[z]['acne_score'] for z in regions_info])
        red_global = np.mean([regions_info[z]['redness'] for z in regions_info])
        oil_global = (
            regions_info['nose']['oiliness']*0.6 +
            regions_info['forehead']['oiliness']*0.2 +
            (regions_info['left_cheek']['oiliness'] + regions_info['right_cheek']['oiliness'])*0.1
        )
        pore_global = np.mean([regions_info[z]['pore_score'] for z in regions_info])

        # summary
        summary = summarize_skin_vi(acne_global, red_global, oil_global, pore_global)
        skin_type = classify_skin_type(acne_global, red_global, oil_global, pore_global)

        # result
        result = {
            "analysis_id": aid,
            "timestamp": now_iso(),
            "scores": {
                "muc_do_mun": acne_global,
                "do_do": red_global,
                "do_dau": oil_global,
                "lo_chan_long": pore_global
            },
            "skin_type": skin_type,
            "summary": summary,
            "zones": {
                z: {
                    "acne_score": info["acne_score"],
                    "acne_count": info["acne_count"],
                    "redness": info["redness"],
                    "oiliness": info["oiliness"],
                    "pore_score": info["pore_score"],
                    "melanin_spots": info["melanin_spots"]
                }
                for z, info in regions_info.items()
            },
            "paths": {
                "original": str(orig_path)
            }
        }

        with open(adir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "analysis_id": aid,
            "chi_so": result["scores"],
            "loai_da": skin_type,
            "tom_tat_tinh_trang_da": summary
        })

    except Exception as e:
        log.exception("Analyze error")
        raise HTTPException(500, f"Internal Error: {e}")


@app.get("/analysis/{analysis_id}/zones")
def get_zones(analysis_id: str):
    adir = BASE_TMP/analysis_id
    rf = adir/"result.json"
    if not rf.exists(): raise HTTPException(404,"analysis_id không tồn tại")
    with open(rf,"r",encoding="utf-8") as f: data=json.load(f)
    return JSONResponse({"analysis_id":analysis_id, "zones": data.get("zones"), "summary": data.get("summary"), "skin_type": data.get("skin_type")})


@app.post("/ai-suggestion")
async def ai_suggestion(payload: dict):
    """
    payload:
    - { "analysis_id": "...", "per_zone": true/false }
    or
    - { "muc_do_mun":..., "do_do":..., "do_dau":..., "lo_chan_long":..., "per_zone": true/false }
    """
    try:
        per_zone = bool(payload.get("per_zone", False))
        if "analysis_id" in payload:
            aid = payload["analysis_id"]; adir = BASE_TMP/aid; rf = adir/"result.json"
            if not rf.exists(): raise HTTPException(404,"analysis_id không tồn tại")
            with open(rf,"r",encoding="utf-8") as f: data=json.load(f)
            scores = data["scores"]
            acne = float(scores["muc_do_mun"]); red = float(scores["do_do"])
            oil = float(scores["do_dau"]); pore = float(scores["lo_chan_long"])
        else:
            acne = float(payload["muc_do_mun"]); red = float(payload["do_do"])
            oil = float(payload["do_dau"]); pore = float(payload["lo_chan_long"])
            aid = uuid.uuid4().hex; adir = make_analysis_dir(aid)

        suggestion = gemini_call_cached(round(acne,3), round(red,3), round(oil,3), round(pore,3), per_zone=per_zone)
        sug_path = adir/"ai_suggestion.txt"
        with open(sug_path,"w",encoding="utf-8") as f: f.write(suggestion)
        return JSONResponse({"goi_y_AI": suggestion, "saved_to": str(sug_path)})
    except Exception as e:
        log.exception("ai suggestion error")
        raise HTTPException(500, f"AI Suggestion Error: {e}")

@app.get("/")
def home(): return {"status":"ok","version":"Skin Analyzer PRO"}
