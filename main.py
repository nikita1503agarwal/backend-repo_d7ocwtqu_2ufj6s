import os
import io
import hmac
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from database import db, create_document
from schemas import Resume, Log, Payment, User  # noqa: F401

import requests

# PDF/DOCX parsing
from PyPDF2 import PdfReader
import zipfile
import re
import html as ihtml

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# Gemini
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Resume Refiner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RefineRequest(BaseModel):
    uid: str
    resume_id: str
    target_role: str

class ReferralRequest(BaseModel):
    uid: str
    referrer_uid: Optional[str] = None

class VerifyPaymentRequest(BaseModel):
    uid: str
    order_id: str
    payment_id: str
    signature: str

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Resume Refiner"}

@app.get("/test")
def test_database():
    return {
        "backend": "running",
        "db": "connected" if db is not None else "not_connected",
        "collections": db.list_collection_names() if db is not None else [],
    }

# Utilities

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    # Pure-Python docx text extraction (no lxml dependency)
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            xml = z.read('word/document.xml').decode('utf-8', errors='ignore')
            # Replace tags with newlines then collapse
            text = re.sub(r'<w:p[\s\S]*?>', '\n', xml)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = ihtml.unescape(text)
            # Normalize whitespace
            lines = [re.sub(r'\s+', ' ', ln).strip() for ln in text.split('\n')]
            lines = [ln for ln in lines if ln]
            return "\n".join(lines).strip()
    except Exception:
        return ""


def generate_pdf_from_text(text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    max_width = width - 2 * margin
    y = height - margin

    # Simple word wrap
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for w in words:
            test = f"{line} {w}".strip()
            if c.stringWidth(test, "Helvetica", 10) <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        lines.append("")

    c.setFont("Helvetica", 10)
    for line in lines:
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        c.drawString(margin, y, line)
        y -= 12

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


@app.post("/api/upload")
async def upload_resume(uid: str = Form(...), target_role: str = Form(...), file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    content = await file.read()
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(content)
    else:
        text = extract_text_from_docx(content)

    resume = Resume(uid=uid, original_text=text, target_role=target_role)
    resume_id = create_document("resume", resume)
    create_document("log", Log(uid=uid, type="upload", message="Uploaded resume", meta={"resume_id": resume_id}).model_dump())
    return {"resume_id": resume_id, "characters": len(text)}


@app.post("/api/refine")
async def refine_resume(req: RefineRequest):
    # Fetch original text
    from bson import ObjectId
    doc = db["resume"].find_one({"_id": ObjectId(req.resume_id), "uid": req.uid})
    if not doc:
        raise HTTPException(status_code=404, detail="Resume not found")
    original = doc.get("original_text", "")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    prompt = f"""
    You are an expert resume writer and ATS optimizer. Rewrite the following resume to target the role: {req.target_role}.
    Requirements:
    - Improve grammar and clarity
    - Use strong action verbs
    - Quantify achievements with numbers where reasonable
    - Ensure ATS-friendly formatting and include relevant keywords for {req.target_role}
    - Use concise bullet points and clear section headings (Summary, Experience, Education, Skills, Projects)
    - Keep overall length to 1 page if possible

    Resume text:
    {original}
    """

    model = genai.GenerativeModel("gemini-pro")
    result = model.generate_content(prompt)
    refined = (getattr(result, 'text', '') or '').strip()

    db["resume"].update_one({"_id": doc["_id"]}, {"$set": {"refined_text": refined, "status": "refined", "target_role": req.target_role, "updated_at": datetime.utcnow()}})
    create_document("log", Log(uid=req.uid, type="refine", message="Refined resume", meta={"resume_id": req.resume_id, "role": req.target_role}).model_dump())

    # return preview info (first 30%)
    preview_len = max(200, int(0.3 * len(refined)))
    return {"refined_preview": refined[:preview_len], "total_length": len(refined)}


@app.get("/api/preview")
async def get_preview(uid: str, resume_id: str):
    from bson import ObjectId
    doc = db["resume"].find_one({"_id": ObjectId(resume_id), "uid": uid})
    if not doc or not doc.get("refined_text"):
        raise HTTPException(status_code=404, detail="Not found")
    text = doc["refined_text"]
    preview_len = max(200, int(0.3 * len(text)))
    return {"refined_preview": text[:preview_len], "total_length": len(text)}


@app.post("/api/razorpay/order")
async def create_razorpay_order(uid: str = Form(...), resume_id: str = Form(...)):
    if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET):
        raise HTTPException(status_code=500, detail="Razorpay keys not configured")
    amount = 4900  # 49 INR in paise
    payload = {
        "amount": amount,
        "currency": "INR",
        "receipt": f"res-{resume_id}",
        "payment_capture": 1
    }
    auth = (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    r = requests.post("https://api.razorpay.com/v1/orders", auth=auth, json=payload, timeout=20)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Failed to create order: {r.text}")
    data = r.json()
    pay = Payment(uid=uid, provider="razorpay", amount=amount, currency="INR", status="created", order_id=data.get("id"))
    _ = create_document("payment", pay)
    create_document("log", Log(uid=uid, type="payment", message="Created Razorpay order", meta={"resume_id": resume_id, "order": data}).model_dump())
    return {"order": data, "key_id": RAZORPAY_KEY_ID}


@app.post("/api/razorpay/verify")
async def verify_razorpay(req: VerifyPaymentRequest):
    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=500, detail="Razorpay secret not configured")
    generated_signature = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        f"{req.order_id}|{req.payment_id}".encode(),
        hashlib.sha256
    ).hexdigest()

    if generated_signature != req.signature:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Mark payment as paid and grant credit (1 download)
    db["payment"].update_one({"order_id": req.order_id}, {"$set": {"status": "paid", "payment_id": req.payment_id, "signature": req.signature}})
    db["user"].update_one({"uid": req.uid}, {"$inc": {"credits": 1}}, upsert=True)
    create_document("log", Log(uid=req.uid, type="payment", message="Payment verified", meta={"order_id": req.order_id}).model_dump())
    return {"status": "verified"}


@app.get("/api/download")
async def download_pdf(uid: str, resume_id: str):
    from bson import ObjectId
    doc = db["resume"].find_one({"_id": ObjectId(resume_id), "uid": uid})
    if not doc or not doc.get("refined_text"):
        raise HTTPException(status_code=404, detail="Not found")

    # Check credits
    user = db["user"].find_one({"uid": uid}) or {"credits": 0}
    if user.get("credits", 0) <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    # Consume a credit
    db["user"].update_one({"uid": uid}, {"$inc": {"credits": -1}})

    pdf_bytes = generate_pdf_from_text(doc["refined_text"])

    create_document("log", Log(uid=uid, type="download", message="Downloaded refined PDF", meta={"resume_id": resume_id}).model_dump())

    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=refined_resume_{resume_id}.pdf"
    })


@app.post("/api/referral")
async def referral(req: ReferralRequest):
    # Save user if not exists
    db["user"].update_one({"uid": req.uid}, {"$setOnInsert": {"email": "", "name": ""}}, upsert=True)

    if req.referrer_uid and req.referrer_uid != req.uid:
        # count this referral once per referred user
        already = db["referral"].find_one({"referrer": req.referrer_uid, "referred": req.uid})
        if not already:
            db["referral"].insert_one({"referrer": req.referrer_uid, "referred": req.uid, "created_at": datetime.utcnow()})
            # increment referrer count
            u = db["user"].find_one_and_update({"uid": req.referrer_uid}, {"$inc": {"referrals_count": 1}}, return_document=True)
            # grant credit for every 3 referrals
            new_count = (u or {}).get("referrals_count", 0) + 1
            if new_count % 3 == 0:
                db["user"].update_one({"uid": req.referrer_uid}, {"$inc": {"credits": 1}})
                create_document("log", Log(uid=req.referrer_uid, type="referral", message="Earned 1 credit for 3 referrals").model_dump())
    return {"ok": True}


@app.get("/api/me")
async def me(uid: str):
    user = db["user"].find_one({"uid": uid}) or {"uid": uid, "credits": 0, "referrals_count": 0}
    return {"uid": uid, "credits": user.get("credits", 0), "referrals_count": user.get("referrals_count", 0)}
