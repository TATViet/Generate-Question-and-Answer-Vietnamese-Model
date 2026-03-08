
### Generate Question and Answer Vietnamese Model

**Official Benchmark Toolkit for Vietnamese Question and Answer Generation (QAG)**

This repository provides the official training code, evaluation scripts, benchmark protocol, and `run_all.sh` for **Vietnamese Question & Answer Generation** on the UIT-ViQuAD 2.0 dataset.

It supports **6 popular Seq2Seq models** and **4 training paradigms**, enabling fair comparison and fully reproducible research.

## 1.Features
- **6 supported models**:
  - `google/mt5-small`
  - `google/mt5-base`
  - `VietAI/vit5-base`
  - `vinai/bartpho-syllable`
  - `vinai/bartpho-word`
  - `facebook/mbart-large-50-many-to-many-mmt`

- **4 training methods**:
  - **pipeline** (traditional two-stage)
  - **multitask** (joint Answer Generation + Question Generation)
  - **end2end** (generate formatted “question: … answer: …” pairs)
  - **instruction** (13 Vietnamese prompt templates — fixed or random mode)

- Comprehensive evaluation: ROUGE-1/2/L, BLEU, chrF, Answer-in-Context accuracy, and QA format consistency
- Automatic job skipping (via `results/metrics.jsonl`)
- NaN checkpoint handling + automatic fp16 → bf16 fallback
- One-command full benchmark with `run_all.sh`

## 2.Dataset
Public dataset on Hugging Face:  
**`taidng/UIT-ViQuAD2.0`**

Only non-impossible questions are used (`is_impossible = False`).

### Quick Installation

```bash
git clone https://github.com/yourusername/Generate-Question-and-Answer-Vietnamese-Model.git
cd Generate-Question-and-Answer-Vietnamese-Model
pip install -r requirements.txt
```

## 3.How to Run

Single experiment

```bash
# Recommended: Instruction tuning with mt5-base
python train_qag_benchmark.py \
  --model_name google/mt5-base \
  --method instruction \
  --instr_mode random \
  --epochs 10 \
  --do_train --do_eval --fp16

# End-to-End with BARTPho
python train_qag_benchmark.py \
  --model_name vinai/bartpho-syllable \
  --method end2end \
  --do_train --do_eval
```

Full benchmark (6 models × all methods)

```bash
# Just one command — automatically skips completed jobs
bash run_all.sh
```

Checkpoints are saved in `checkpoints/`.  
All results are automatically appended to:
- `results/metrics.csv`
- `results/metrics.jsonl`

## 4.Evaluation Metrics
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BLEU (1–4 gram precisions + brevity penalty)
- chrF
- **Answer-in-Context** (% of generated answers that appear in the context)
- **QA format OK** (for end2end & instruction methods only)

## 5.Intended Use
This toolkit is intended for:
- Research on Vietnamese Question Generation & Answer Generation
- Comparing Seq2Seq architectures for Vietnamese
- Building Vietnamese QA systems and chatbots
- Academic and non-commercial research only

**Not intended for commercial use.**

## 6.Limitations
- Performance scales with model size (base models usually outperform small)
- Instruction tuning quality depends on the 13 provided templates
- Results are specific to UIT-ViQuAD 2.0 and may differ on other Vietnamese domains

## 7.Citation
If you use this codebase, the benchmark protocol, or any trained models, please cite:

```bibtex
@misc{generateqa_vietnamese2025,
  title={Generate Question and Answer Vietnamese Model},
  author={Trần},
  year={2025},
  url={https://github.com/TATViet/Generate-Question-and-Answer-Vietnamese-Model}
}
```

## 8.Contact
- **Trần**  
- Email: trananhtracviet20052011@gmail.com

