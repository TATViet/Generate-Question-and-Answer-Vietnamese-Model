import os
import re
import json
import csv
import time
import random
import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

def _ensure_evaluate_installed():
    try:
        import evaluate as _evaluate  # noqa: F401
        return
    except ModuleNotFoundError:
        print("Missing dependency 'evaluate'. Installing required metric packages...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "evaluate",
            "rouge_score",
            "sacrebleu",
        ])


_ensure_evaluate_installed()
import evaluate


INSTRUCTIONS = [
    "Đặt ra một số câu hỏi và câu trả lời cho đoạn văn sau.",
    "Tạo ra một vài cặp câu hỏi và câu trả lời tương ứng với đoạn văn sau.",
    "Tạo ra một số câu hỏi và câu trả lời của chúng dựa trên văn bản sau.",
    "Xây dựng câu hỏi và câu trả lời từ đoạn văn đã cho.",
    "Phát triển một tập hợp cặp câu hỏi-câu trả lời cho văn bản bên dưới.",
    "Xây dựng câu hỏi cùng với câu trả lời của chúng cho nội dung sau.",
    "Sản xuất cặp câu hỏi-câu trả lời được lấy từ đoạn văn.",
    "Nghĩ ra một số câu hỏi và câu trả lời liên quan đến đoạn văn sau.",
    "Xây dựng danh sách câu hỏi và câu trả lời cho văn bản đã cho.",
    "Tạo ra các kết hợp câu hỏi-câu trả lời dựa trên đoạn văn được cung cấp.",
    "Tạo ra các cặp QA cho đoạn văn sau.",
    "Xây dựng một số Q&A từ văn bản bên dưới.",
    "Phát triển câu hỏi và câu trả lời tương ứng với đoạn văn.",
]


# -----------------------------
# Helpers
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_answer_in_context(answer: str, context: str) -> int:
    if not answer:
        return 0
    ans = normalize_text(answer).lower()
    ctx = normalize_text(context).lower()
    return 1 if ans in ctx else 0

QA_PARSE_RE = re.compile(
    r"question\s*:\s*(?P<q>.*?)(?:\s+answer\s*:\s*(?P<a>.*))?$",
    flags=re.IGNORECASE | re.DOTALL
)

def parse_qa_from_text(gen_text: str) -> Tuple[str, str]:
    """
    Parse string formatted like:
      "question: ... answer: ..."
    Return (question, answer). If fail: (whole, "").
    """
    t = normalize_text(gen_text)
    m = QA_PARSE_RE.match(t)
    if not m:
        return (t, "")
    q = normalize_text(m.group("q") or "")
    a = normalize_text(m.group("a") or "")
    return (q, a)

def build_output_dir(base_dir: str, model_name: str, method: str, instr_mode: str) -> str:
    model_short = model_name.split("/")[-1].replace("-", "_")
    if method == "instruction":
        return os.path.join(base_dir, f"{model_short}_{method}_{instr_mode}_ckpt")
    return os.path.join(base_dir, f"{model_short}_{method}_ckpt")


def checkpoint_has_nan(checkpoint_dir: str) -> bool:
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return False
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        for item in state.get("log_history", []):
            for k in ("loss", "eval_loss", "grad_norm"):
                v = item.get(k)
                if isinstance(v, float) and math.isnan(v):
                    return True
    except Exception:
        return False
    return False


# -----------------------------
# Preprocess functions
# -----------------------------
def preprocess_pipeline(examples, tokenizer, max_src_len: int, max_tgt_len: int):
    inputs = [
        "generate question: " + c + " answer: " + a["text"][0]
        for c, a in zip(examples["context"], examples["answers"])
    ]
    targets = examples["question"]

    model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_tgt_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_multitask(examples, tokenizer, max_src_len: int, max_tgt_len: int):
    inputs, targets = [], []
    for c, q, a in zip(examples["context"], examples["question"], examples["answers"]):
        a0 = a["text"][0]
        inputs.append("generate answer: " + c)
        targets.append(a0)
        inputs.append("generate question: " + c + " answer: " + a0)
        targets.append(q)

    model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_tgt_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_end2end(examples, tokenizer, max_src_len: int, max_tgt_len: int):
    inputs = ["generate qa: " + c for c in examples["context"]]
    targets = [
        "question: " + q + " answer: " + a["text"][0]
        for q, a in zip(examples["question"], examples["answers"])
    ]

    model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_tgt_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_instruction(
    examples, tokenizer, max_src_len: int, max_tgt_len: int,
    instr_mode: str, instr_fixed_id: int, seed: int
):
    rng = random.Random(seed)

    def pick_instruction() -> str:
        if instr_mode == "fixed":
            idx = max(0, min(instr_fixed_id, len(INSTRUCTIONS) - 1))
            return INSTRUCTIONS[idx]
        return rng.choice(INSTRUCTIONS)

    inputs = [pick_instruction() + " " + c for c in examples["context"]]
    targets = [
        "question: " + q + " answer: " + a["text"][0]
        for q, a in zip(examples["question"], examples["answers"])
    ]

    model_inputs = tokenizer(inputs, max_length=max_src_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_tgt_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# -----------------------------
# Evaluation
# -----------------------------
@dataclass
class EvalConfig:
    method: str
    max_gen_len: int
    num_beams: int


def compute_text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    preds = [normalize_text(x) for x in preds]
    refs = [normalize_text(x) for x in refs]

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=False)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_res = chrf.compute(predictions=preds, references=refs)
    precisions = bleu_res.get("precisions", [float("nan")] * 4)
    if not isinstance(precisions, list):
        precisions = [float("nan")] * 4
    precisions = list(precisions[:4]) + [float("nan")] * max(0, 4 - len(precisions))

    return {
        "rouge1": float(rouge_res.get("rouge1", float("nan"))),
        "rouge2": float(rouge_res.get("rouge2", float("nan"))),
        "rougeL": float(rouge_res["rougeL"]),
        "bleu": float(bleu_res["bleu"]),
        "bleu_p1": float(precisions[0]),
        "bleu_p2": float(precisions[1]),
        "bleu_p3": float(precisions[2]),
        "bleu_p4": float(precisions[3]),
        "bleu_bp": float(bleu_res.get("brevity_penalty", float("nan"))),
        "bleu_len_ratio": float(bleu_res.get("length_ratio", float("nan"))),
        "bleu_pred_len": float(bleu_res.get("translation_length", float("nan"))),
        "bleu_ref_len": float(bleu_res.get("reference_length", float("nan"))),
        "chrf": float(chrf_res["score"]),
    }


def prefix_metrics(prefix: str, m: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in m.items()}


def empty_prefixed_metrics(prefix: str) -> Dict[str, float]:
    keys = [
        "rouge1",
        "rouge2",
        "rougeL",
        "bleu",
        "bleu_p1",
        "bleu_p2",
        "bleu_p3",
        "bleu_p4",
        "bleu_bp",
        "bleu_len_ratio",
        "bleu_pred_len",
        "bleu_ref_len",
        "chrf",
    ]
    return {f"{prefix}_{k}": float("nan") for k in keys}


def sanitize_prediction_ids(pred_ids, tokenizer) -> np.ndarray:
    arr = np.asarray(pred_ids)

    # Some trainer/version combinations can return logits [bs, seq, vocab].
    if arr.ndim == 3:
        arr = np.argmax(arr, axis=-1)

    # Guard against NaN/Inf and out-of-range values before decode.
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.rint(arr)

    arr = arr.astype(np.int64, copy=False)

    vocab_size = None
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    except Exception:
        vocab_size = 0

    if vocab_size > 0:
        arr = np.clip(arr, 0, vocab_size - 1)
    else:
        arr = np.clip(arr, 0, np.iinfo(np.int32).max)

    return arr


def evaluate_model(trainer: Seq2SeqTrainer, tokenizer: AutoTokenizer, raw_val, eval_cfg: EvalConfig) -> Dict[str, float]:
    pred_output = trainer.predict(
        trainer.eval_dataset,
        max_length=eval_cfg.max_gen_len,
        num_beams=eval_cfg.num_beams,
    )

    pred_ids = pred_output.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pred_ids = sanitize_prediction_ids(pred_ids, tokenizer)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    preds = [normalize_text(x) for x in preds]

    method = eval_cfg.method

    if method == "multitask":
        # evaluate only QG half for fairness
        preds_qg = preds[1::2]
        refs_qg = [normalize_text(q) for q in raw_val["question"]]
        n = min(len(preds_qg), len(refs_qg))
        m = compute_text_metrics(preds_qg[:n], refs_qg[:n])
        return {
            **prefix_metrics("q", m),
            **empty_prefixed_metrics("a"),
            "a_in_ctx": float("nan"),
            "qa_format_ok": float("nan"),
        }

    if method == "pipeline":
        refs_q = [normalize_text(q) for q in raw_val["question"]]
        n = min(len(preds), len(refs_q))
        m = compute_text_metrics(preds[:n], refs_q[:n])
        return {
            **prefix_metrics("q", m),
            **empty_prefixed_metrics("a"),
            "a_in_ctx": float("nan"),
            "qa_format_ok": float("nan"),
        }

    # end2end / instruction
    gold_q = [normalize_text(q) for q in raw_val["question"]]
    gold_a = [normalize_text(a["text"][0]) for a in raw_val["answers"]]
    contexts = list(raw_val["context"])

    parsed_q, parsed_a, format_ok = [], [], []
    for t in preds:
        q, a = parse_qa_from_text(t)
        parsed_q.append(q)
        parsed_a.append(a)
        format_ok.append(1 if q else 0)

    n = min(len(parsed_q), len(gold_q), len(gold_a), len(contexts))
    parsed_q, parsed_a = parsed_q[:n], parsed_a[:n]
    gold_q, gold_a = gold_q[:n], gold_a[:n]
    contexts = contexts[:n]
    format_ok = format_ok[:n]

    qm = compute_text_metrics(parsed_q, gold_q)
    am = compute_text_metrics(parsed_a, gold_a)

    a_in_ctx = [contains_answer_in_context(pa, ctx) for pa, ctx in zip(parsed_a, contexts)]

    return {
        **prefix_metrics("q", qm),
        **prefix_metrics("a", am),
        "a_in_ctx": float(np.mean(a_in_ctx)) if len(a_in_ctx) else 0.0,
        "qa_format_ok": float(np.mean(format_ok)) if len(format_ok) else 0.0,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="taidng/UIT-ViQuAD2.0")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--method", type=str, required=True,
                        choices=["pipeline", "multitask", "end2end", "instruction"])
    parser.add_argument("--instr_mode", type=str, default="na",
                        choices=["na", "fixed", "random"])
    parser.add_argument("--instr_fixed_id", type=int, default=0)

    parser.add_argument("--output_base", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=8)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=128)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_base, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)
    dataset["train"] = dataset["train"].filter(lambda x: not x["is_impossible"])
    dataset["validation"] = dataset["validation"].filter(lambda x: not x["is_impossible"])
    raw_val = dataset["validation"]

    tok_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # mT5/T5 is prone to NaN overflow in fp16; prefer bf16 when available.
    model_type = str(getattr(model.config, "model_type", "")).lower()
    use_fp16 = bool(args.fp16)
    use_bf16 = bool(args.bf16)
    if use_fp16 and model_type in {"t5", "mt5", "umt5"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print(
                f"[precision] {model_type} + fp16 can produce NaN. "
                "Switching to bf16 automatically."
            )
            use_fp16 = False
            use_bf16 = True
        else:
            print(
                f"[precision] {model_type} + fp16 can produce NaN. "
                "Disabling fp16 and using fp32."
            )
            use_fp16 = False

    if args.method == "instruction" and args.instr_mode == "na":
        raise ValueError("method=instruction requires --instr_mode fixed or random")

    output_dir = build_output_dir(args.output_base, args.model_name, args.method, args.instr_mode)
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None
    resume_checkpoint = last_checkpoint
    if resume_checkpoint and checkpoint_has_nan(resume_checkpoint):
        print(
            f"[resume] Detected NaN in {resume_checkpoint}/trainer_state.json. "
            "Ignoring checkpoint and starting fresh."
        )
        resume_checkpoint = None

    # Tokenize
    if args.method == "pipeline":
        tokenized_train = dataset["train"].map(
            lambda x: preprocess_pipeline(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["train"].column_names
        )
        tokenized_val = dataset["validation"].map(
            lambda x: preprocess_pipeline(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["validation"].column_names
        )

    elif args.method == "multitask":
        tokenized_train = dataset["train"].map(
            lambda x: preprocess_multitask(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["train"].column_names
        )
        tokenized_val = dataset["validation"].map(
            lambda x: preprocess_multitask(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["validation"].column_names
        )

    elif args.method == "end2end":
        tokenized_train = dataset["train"].map(
            lambda x: preprocess_end2end(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["train"].column_names
        )
        tokenized_val = dataset["validation"].map(
            lambda x: preprocess_end2end(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True, remove_columns=dataset["validation"].column_names
        )

    else:  # instruction
        tokenized_train = dataset["train"].map(
            lambda x: preprocess_instruction(
                x, tokenizer, args.max_src_len, args.max_tgt_len,
                args.instr_mode, args.instr_fixed_id, args.seed
            ),
            batched=True, remove_columns=dataset["train"].column_names
        )
        tokenized_val = dataset["validation"].map(
            lambda x: preprocess_instruction(
                x, tokenizer, args.max_src_len, args.max_tgt_len,
                args.instr_mode, args.instr_fixed_id, args.seed
            ),
            batched=True, remove_columns=dataset["validation"].column_names
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    use_early_stopping = args.early_stopping_patience > 0
    save_strategy = "epoch" if use_early_stopping else "steps"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=args.wd,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=use_fp16,
        bf16=use_bf16,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        logging_strategy="steps",
        logging_steps=50,
        logging_nan_inf_filter=False,
        report_to="none",
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=resume_checkpoint,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    t0 = time.time()

    if args.do_train:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    metrics = {}
    if args.do_eval:
        eval_cfg = EvalConfig(method=args.method, max_gen_len=args.max_gen_len, num_beams=args.num_beams)
        metrics = evaluate_model(trainer, tokenizer, raw_val, eval_cfg)

    elapsed = time.time() - t0

    row = {
        "model_name": args.model_name,
        "tokenizer_name": tok_name,
        "method": args.method,
        "instr_mode": args.instr_mode,
        "instr_fixed_id": args.instr_fixed_id if (args.method == "instruction" and args.instr_mode == "fixed") else -1,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "fp16": int(use_fp16),
        "bf16": int(use_bf16),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "early_stopping_enabled": int(use_early_stopping),
        "num_beams": args.num_beams,
        "max_gen_len": args.max_gen_len,
        "max_src_len": args.max_src_len,
        "max_tgt_len": args.max_tgt_len,
        "output_dir": output_dir,
        "elapsed_sec": elapsed,
        **metrics,
    }

    # JSONL append
    jsonl_path = os.path.join(args.results_dir, "metrics.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV append
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("Saved results to:", csv_path, "and", jsonl_path)
    print("Row:", row)


if __name__ == "__main__":
    main()
