# app.py - Skin Analyzer LITE v6 (no torch, no mediapipe — deployable on Railway)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import cv2, os, uuid, io, json, datetime, time
from PIL import Image
import google.generativeai as genai

# -------------------------
# Setup
# -------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

app = FastAPI(title="Skin Analyzer Lite v6")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_TMP = Path("tmp")
BASE_TMP.mkdir(exist_ok=True, parents=True)

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def make_dir(aid):
    p = BASE_TMP / aid
    p.mkdir(exist_ok=True)
    return p

# --------------------------------
# ANALYSIS HELPERS (no ML)
# --------------------------------
def acne_score(np_bgr, mask):
    ycrcb = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2YCrCb)
    cr = cv2.bitwise_and(ycrcb[:,:,1], ycrcb[:,:,1], mask=mask)
    blur = cv2.GaussianBlur(cr, (5,5), 0)
    if mask.sum() == 0: return 0, 0
    norm = cv2.normalize(blur, None, 0,255,cv2.NORM_MINMAX)
    thresh = cv2.adaptiveThreshold(np.uint8(norm),255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,15,3)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = [c for c in cnts if 8 < cv2.contourArea(c) < 1000]
    return min(1.0, len(spots)/50), len(spots)

def redness(np_bgr, mask):
    lab = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    a = cv2.bitwise_and(lab[:,:,1], lab[:,:,1], mask=mask)
    vals = a[a>0]
    if len(vals)==0: return 0
    v = (vals.mean() - 128)/50
    return float(max(0,min(1,v)))

def oiliness(np_bgr, mask):
    hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.bitwise_and(hsv[:,:,2], hsv[:,:,2], mask=mask)
    if mask.sum()==0: return 0
    pct = (v>200).sum()/(mask.sum()/255)
    return float(min(1,pct*3))

def pore(np_bgr, mask):
    gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bitwise_and(gray, gray, mask=mask)
    if mask.sum()==0: return 0
    dog = cv2.absdiff(
        cv2.GaussianBlur(g,(3,3),0),
        cv2.GaussianBlur(g,(11,11),0)
    )
    return float(min(1, np.mean(dog)/20))

def bbox_mask(shape, box):
    h,w = shape[:2]
    x,y,ww,hh = box
    x1=int(x*w); y1=int(y*h); x2=int((x+ww)*w); y2=int((y+hh)*h)
    m = np.zeros((h,w),dtype=np.uint8)
    cv2.rectangle(m,(x1,y1),(x2,y2),255,-1)
    return m

FALLBACK_ZONES = {
    "forehead": (0.2,0.05,0.6,0.18),
    "nose": (0.38,0.25,0.24,0.25),
    "left_cheek": (0.05,0.25,0.35,0.4),
    "right_cheek":(0.6,0.25,0.35,0.4),
    "chin": (0.3,0.65,0.4,0.25)
}

# -------------------------
# SUMMARY + SKIN TYPE
# -------------------------
def summarize(ac, rd, oil, pore_):
    out=[]
    out.append("Da có nhiều mụn." if ac>0.6 else "Da có mụn nhẹ." if ac>0.3 else "Da ít mụn.")
    out.append("Da dễ đỏ, nhạy cảm." if rd>0.3 else "Độ đỏ thấp.")
    out.append("Da dầu nhiều." if oil>0.5 else "Da hơi dầu." if oil>0.3 else "Da khô – bình thường.")
    out.append("Lỗ chân lông to." if pore_>0.4 else "Lỗ chân lông nhỏ – trung bình.")
    return " ".join(out)

def skin_type(ac, rd, oil, pore_):
    if rd>0.35: return "Da nhạy cảm"
    if oil>0.45 and pore_>0.35: return "Da dầu – lỗ chân lông to"
    if oil>0.35: return "Da dầu"
    if oil<0.2 and ac<0.3: return "Da khô – bình thường"
    return "Da hỗn hợp"

# -------------------------
# ENDPOINT: ANALYZE
# -------------------------
@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    aid = uuid.uuid4().hex
    adir = make_dir(aid)

    # save original
    img_path = adir/"orig.jpg"
    with open(img_path,"wb") as f: f.write(data)

    img = Image.open(io.BytesIO(data)).convert("RGB")
    np_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    zones={}
    for z,box in FALLBACK_ZONES.items():
        mask = bbox_mask(np_bgr.shape, box)
        ac,ac_n = acne_score(np_bgr, mask)
        rd = redness(np_bgr, mask)
        oi = oiliness(np_bgr, mask)
        pr = pore(np_bgr, mask)
        zones[z]={
            "acne_score": ac,
            "acne_count": ac_n,
            "redness": rd,
            "oiliness": oi,
            "pore_score": pr
        }

    # global
    ac_g = np.mean([zones[z]['acne_score'] for z in zones])
    rd_g = np.mean([zones[z]['redness'] for z in zones])
    oil_g = zones['nose']['oiliness']*0.6 + zones['forehead']['oiliness']*0.2 + \
            (zones['left_cheek']['oiliness']+zones['right_cheek']['oiliness'])*0.1
    pr_g = np.mean([zones[z]['pore_score'] for z in zones])

    summary = summarize(ac_g, rd_g, oil_g, pr_g)
    s_type = skin_type(ac_g, rd_g, oil_g, pr_g)

    result={
        "analysis_id": aid,
        "timestamp": now_iso(),
        "scores":{
            "muc_do_mun": ac_g,
            "do_do": rd_g,
            "do_dau": oil_g,
            "lo_chan_long": pr_g
        },
        "skin_type": s_type,
        "summary": summary,
        "zones": zones
    }

    with open(adir/"result.json","w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)

    return JSONResponse({
        "analysis_id": aid,
        "chi_so": result["scores"],
        "loai_da": s_type,
        "tom_tat_tinh_trang_da": summary
    })

# -------------------------
# ENDPOINT: AI SUGGESTION
# -------------------------
@app.post("/ai-suggestion")
async def ai_suggestion(payload: dict):
    if not GEMINI_KEY:
        return {"goi_y_AI": "Gemini chưa được cấu hình."}

    per_zone = payload.get("per_zone", False)

    if "analysis_id" in payload:
        aid = payload["analysis_id"]
        path = BASE_TMP/aid/"result.json"
        if not path.exists():
            raise HTTPException(404,"analysis_id không tồn tại")
        data = json.load(open(path,"r",encoding="utf-8"))
        sc = data["scores"]
        ac, rd, oil, pr = sc["muc_do_mun"], sc["do_do"], sc["do_dau"], sc["lo_chan_long"]
    else:
        ac = payload["muc_do_mun"]
        rd = payload["do_do"]
        oil = payload["do_dau"]
        pr = payload["lo_chan_long"]

    prompt = f"Bạn là chuyên gia da liễu. Giá trị: mụn={ac}, đỏ={rd}, dầu={oil}, lcl={pr}. "
    if per_zone: prompt += "Hãy gợi ý theo từng vùng mặt. "
    prompt += "Trả lời ngắn gọn, rõ ràng, có bước routine sáng – tối."

    res = genai.GenerativeModel("models/gemini-2.0-flash-lite").generate_content(prompt)
    text = getattr(res,"text",None)
    return {"goi_y_AI": text or "AI không phản hồi"}

@app.get("/")
def home():
    return {"status":"ok","v":"skin-ai"}
