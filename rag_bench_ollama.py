#!/usr/bin/env python3
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
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

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# -----------------------------
# Utils: metrics (EM/F1, recall)
# -----------------------------
_ws = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    return _ws.sub(" ", s.strip().lower())

def thai_tokens(s: str) -> List[str]:
    # ใช้ newmm (ดีพอสำหรับ F1 เบื้องต้น)
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
    # XQuAD Thai ใช้ split 'validation' ชุดเดียว ~1,190 ตัวอย่าง
    from datasets import load_dataset
    return load_dataset("google/xquad", "xquad.th", split="validation")  # <- สำคัญ!


# -----------------------------
# Chunking & Embeddings & Index
# -----------------------------
def build_chunks_from_contexts(contexts: List[str], target_len=700, overlap=120) -> (List[str], List[int]):
    """Return chunks list + mapping to source doc index."""
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

def build_index(chunks: List[str], model_name="BAAI/bge-m3"):
    emb = SentenceTransformer(model_name)
    vecs = emb.encode(chunks, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
    return emb, index

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
def rag_ask(question: str, k: int, emb, index, chunks: List[str]):
    qv = emb.encode([question], normalize_embeddings=True, convert_to_numpy=True, normalize_to_unit=True)
    D, I = index.search(qv, k)
    ctxs = [chunks[i] for i in I[0]]
    ctx = "\n\n".join([f"[{j+1}] {c}" for j, c in enumerate(ctxs)])
    prompt = (
        "คุณเป็นผู้ช่วยตอบคำถามโดยใช้ข้อมูลอ้างอิงเท่านั้น ถ้าไม่พบคำตอบให้บอกว่าไม่ทราบ\n"
        f"ข้อมูลอ้างอิง:\n{ctx}\n\n"
        f"คำถาม: {question}\n"
        "คำตอบเป็นภาษาไทย กระชับ ถูกต้อง และอ้างอิงเฉพาะข้อมูลข้างต้นเท่านั้น:"
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
        gold_ctx = row["context"]

        prompt, retrieved_ctxs = rag_ask(q, k, emb, index, chunks)
        pred = ollama_generate(model_name, prompt)

        # metrics
        em, f1 = em_and_f1(pred, golds)

        # retrieval recall@k: มี "คำตอบจริงอย่างน้อยหนึ่งคำ" อยู่ในเอกสารที่ดึงมาหรือไม่
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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:1b","gemma:2b-instruct","qwen3:1.7b", "scb10x/typhoon-ocr-7b:latest"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=200, help="จำนวนตัวอย่างที่จะทดสอบ")
    ap.add_argument("--embed", type=str, default="BAAI/bge-m3")
    args = ap.parse_args()

    print("โหลดชุดข้อมูล thaiqa_squad ...")
    ds = load_thai_split()


    # สร้างคลังเอกสารจาก context ทั้งหมด (unique)
    contexts = list({r["context"] for r in ds})
    print(f"contexts: {len(contexts)}")

    print("ตัดชิ้นส่วน (chunking) ...")
    chunks, chunk_src = build_chunks_from_contexts(contexts, target_len=700, overlap=120)
    print(f"chunks: {len(chunks)}")

    print(f"ฝังเอกสารด้วย {args.embed} และสร้าง FAISS index ...")
    emb, index = build_index(chunks, model_name=args.embed)

    all_summaries = []
    os.makedirs("runs", exist_ok=True)

    for m in args.models:
        df, summ = eval_model(m, ds, emb, index, chunks, chunk_src, contexts, k=args.k, n=args.n)
        out_csv = f"runs/results_{m.replace(':','_')}.csv"
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
