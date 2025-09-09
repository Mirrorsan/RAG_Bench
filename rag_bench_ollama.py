#!/usr/bin/env python3
# หากตั้งค่านี้จะปิด GPU: os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os, sys, json, time, argparse, re, math, hashlib
from typing import List, Dict
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pythainlp.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import torch  # NEW

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def model_to_fname(m: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', m)

# -----------------------------
# Utils: metrics (EM/F1, recall)
# -----------------------------
_ws = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    return _ws.sub(" ", s.strip().lower())

def thai_tokens(s: str) -> List[str]:
    toks = word_tokenize(s.strip(), engine="newmm")
    return [t.strip() for t in toks if t.strip()]

def em_and_f1(pred: str, golds: List[str]) -> (float, float):
    pred_n = normalize_text(pred)
    golds_n = [normalize_text(g) for g in golds]
    em = 1.0 if pred_n in golds_n else 0.0

    ptoks = set(thai_tokens(pred_n))
    if not ptoks:
        return em, 0.0
    def f1_single(g):
        gtoks = set(thai_tokens(g))
        common = len(ptoks & gtoks)
        if common == 0: return 0.0
        prec = common / max(1, len(ptoks))
        rec  = common / max(1, len(gtoks))
        return 2*prec*rec/(prec+rec)
    f1 = max(f1_single(g) for g in golds_n)
    return em, f1

# -----------------------------
# Load ThaiQA (train/dev)
# -----------------------------
def load_thai_split():
    return load_dataset("google/xquad", "xquad.th", split="validation")

# -----------------------------
# Chunking & Embeddings & Index
# -----------------------------
def build_chunks_from_contexts(contexts: List[str], target_len=700, overlap=120) -> (List[str], List[int]):
    chunks, src = [], []
    for idx, d in enumerate(contexts):
        sents = sent_tokenize(d)
        cur, buf = 0, []
        for s in sents:
            buf.append(s); cur += len(s)
            if cur >= target_len:
                chunks.append(" ".join(buf)); src.append(idx)
                # overlap
                keep = []
                tot = 0
                for ss in reversed(buf):
                    keep.insert(0, ss); tot += len(ss)
                    if tot >= overlap: break
                buf, cur = keep, sum(len(x) for x in keep)
        if buf:
            chunks.append(" ".join(buf)); src.append(idx)
    return chunks, src

def build_embeddings_gpu(chunks: List[str], model_name="BAAI/bge-m3", batch_size=128):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: torch.cuda.is_available() == False")
    device = "cuda"
    emb = SentenceTransformer(model_name, device=device)
    # เข้ารหัสบน GPU
    vecs = emb.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
        device=device,
    )
    return emb, vecs

def build_faiss_gpu_index(vecs: np.ndarray):
    if faiss.get_num_gpus() <= 0:
        raise RuntimeError("FAISS GPU unavailable: faiss.get_num_gpus() == 0 (ติดตั้ง faiss-gpu แล้วหรือยัง?)")
    res = faiss.StandardGpuResources()
    d = vecs.shape[1]
    index = faiss.GpuIndexFlatIP(res, d)
    index.add(vecs.astype(np.float32))
    return index

# -----------------------------
# Ollama inference
# -----------------------------
def ollama_generate(model: str, prompt: str, num_ctx=8192, num_predict=256, temperature=0.0, top_p=1.0, timeout=120):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p
        },
        "stream": False
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        return f"[ERROR calling Ollama: {e}]"

# -----------------------------
# Naive RAG ask()
# -----------------------------
def rag_ask(question: str, k: int, emb: SentenceTransformer, index, chunks: List[str]):
    # เข้ารหัส query บน GPU เช่นกัน
    qv = emb.encode([question], normalize_embeddings=True, convert_to_numpy=True, device="cuda")
    D, I = index.search(qv, k)
    ctxs = [chunks[i] for i in I[0]]
    ctx = "\n\n".join([f"[{j+1}] {c}" for j, c in enumerate(ctxs)])
    prompt = (
        "### บทบาท ###\n"
        "คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้านการค้นหาและสรุปข้อมูล หน้าที่ของคุณคือการตอบคำถามโดยอ้างอิงจาก \"บริบท\" ที่ให้มาอย่างเคร่งครัด\n\n"
        "### ข้อบังคับ ###\n"
        "1.  ตอบคำถามโดยใช้ข้อมูลจาก \"บริบท\" เท่านั้น\n"
        "2.  ห้ามใช้ความรู้ภายนอกหรือความรู้ส่วนตัวในการตอบโดยเด็ดขาด\n"
        "3.  หากข้อมูลในบริบทไม่เพียงพอที่จะตอบคำถามได้ ให้ตอบว่า \"ไม่สามารถหาคำตอบได้จากข้อมูลที่ให้มา\"\n"
        "4.  ตอบเป็นภาษาไทยให้กระชับและตรงประเด็นที่สุด\n\n"
        "### บริบท ###\n"
        f"{ctx}\n\n"
        "### คำถาม ###\n"
        f"{question}\n\n"
        "### คำตอบ ###"
    )
    return prompt, ctxs

# -----------------------------
# Evaluate one model
# -----------------------------
def eval_model(model_name: str, ds, emb, index, chunks, chunk_src_map, contexts, k=5, n=200):
    rows = []
    n = min(n, len(ds))
    for i in tqdm(range(n), desc=f"Eval {model_name}"):
        row = ds[i]
        q = row["question"]
        golds = row["answers"]["text"] if isinstance(row["answers"]["text"], list) else [row["answers"]["text"]]
        prompt, retrieved_ctxs = rag_ask(q, k, emb, index, chunks)
        pred = ollama_generate(model_name, prompt)

        em, f1 = em_and_f1(pred, golds)

        found = False
        for c in retrieved_ctxs:
            for g in golds:
                if normalize_text(g) in normalize_text(c):
                    found = True; break
            if found: break
        recall = 1.0 if found else 0.0

        rows.append({
            "idx": i,
            "model": model_name,
            "question": q,
            "pred": pred,
            "gold": golds[0] if golds else "",
            "EM": em, "F1": f1, "Recall@k": recall
        })
    df = pd.DataFrame(rows)
    summary = df[["EM","F1","Recall@k"]].mean().to_dict()
    return df, summary

# -----------------------------
# Diagnostics
# -----------------------------
def print_gpu_status():
    print("==== GPU STATUS ====")
    try:
        print("PyTorch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Torch device:", torch.cuda.get_device_name(0))
            print("torch.version.cuda:", torch.version.cuda)
    except Exception as e:
        print("Torch check error:", e)
    try:
        print("FAISS #GPUs:", faiss.get_num_gpus())
    except Exception as e:
        print("FAISS GPU check error:", e)
    print("====================")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:1b","gemma:2b-instruct","qwen3:1.7b","scb10x/typhoon-ocr-7b:latest"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=200, help="จำนวนตัวอย่างที่จะทดสอบ")
    ap.add_argument("--embed", type=str, default="BAAI/bge-m3")
    ap.add_argument("--embed-batch", type=int, default=128)
    args = ap.parse_args()

    print_gpu_status()

    print("โหลดชุดข้อมูล thaiqa_squad ...")
    ds = load_thai_split()

    contexts = list({r["context"] for r in ds})
    print(f"contexts: {len(contexts)}")

    print("ตัดชิ้นส่วน (chunking) ...")
    chunks, chunk_src = build_chunks_from_contexts(contexts, target_len=700, overlap=120)
    print(f"chunks: {len(chunks)}")

    print(f"ฝังเอกสารด้วย {args.embed} (GPU) ...")
    emb, vecs = build_embeddings_gpu(chunks, model_name=args.embed, batch_size=args.embed_batch)

    print("สร้าง FAISS GPU index ...")
    index = build_faiss_gpu_index(vecs)

    all_summaries = []
    os.makedirs("runs", exist_ok=True)

    for m in args.models:
        df, summ = eval_model(m, ds, emb, index, chunks, chunk_src, contexts, k=args.k, n=args.n)
        safe_name = model_to_fname(m)
        out_csv = f"runs/results_{safe_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[{m}] summary: {summ}  -> saved: {out_csv}")
        summ["model"] = m
        all_summaries.append(summ)

    sumdf = pd.DataFrame(all_summaries)[["model","EM","F1","Recall@k"]].sort_values("F1", ascending=False)
    sumdf.to_csv("runs/summary.csv", index=False)
    print("\n== Summary ==")
    print(sumdf.to_string(index=False))

if __name__ == "__main__":
    main()
