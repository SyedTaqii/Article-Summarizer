# Article-Summarizer

Fine-tuning T5 for text summarization

This repository contains code and a fine-tuned T5 model for generating concise, abstractive summaries of long news articles.

The demo app is deployed on Streamlit Community Cloud (replace the URL below with your app's URL):

https://article-summarizer-taqi-tallal.streamlit.app/

## 1. Project overview

This project fine-tunes a `t5-small` model (≈60M parameters) on the CNN/DailyMail summarization dataset to produce abstractive summaries.

- Model: `t5-small`
- Dataset: CNN/DailyMail
- Fine-tuned model (example): `syedtaqi/t5-summarizer` (replace with your repo or HF model path)

## 2. Objective

Fine-tune an encoder–decoder (T5) for a sequence-to-sequence summarization task. T5 is a text-to-text model, so inputs are formatted with a task prefix (see Methodology).

## 3. Methodology

- Preprocessing: every article input was prefixed with `summarize: ` so the model knows the task.
- Training: used `Seq2SeqTrainer` with the Adafactor optimizer and learning rate 1e-4 on a 100k-article sample.
- Evaluation: `load_best_model_at_end=True`, `metric_for_best_model="rouge1"` (best model chosen by Rouge-1).

## 4. Quantitative results

The model was trained for 3 epochs and achieved peak performance at the end of epoch 1. The final saved checkpoint corresponds to that epoch.

| Metric   | Score   | Interpretation |
|----------|---------|----------------|
| Rouge-1  | 0.4270  | 42.7% word overlap with reference (strong) |
| Rouge-2  | 0.3263  | 32.6% two-word phrase overlap (very strong) |
| Rouge-L  | 0.3328  | 33.3% longest-common-subsequence overlap |

### Analysis: lead bias

The model performs well, but it shows a "lead bias": it often copies the first 1–3 sentences of an article. This behavior is expected because many news articles place the most important information at the top (inverted pyramid writing style), which the model learns as an effective shortcut for high ROUGE scores.

## 5. How to run this app locally

This app loads a fine-tuned model (from disk or the Hugging Face Hub) and exposes a simple Streamlit UI.

1. Clone the repository

```powershell
git clone https://github.com/SyedTaqii/Article-Summarizer.git
cd Article-Summarizer
```

2. Install dependencies

Required packages (example `requirements.txt`):

```
streamlit
torch
transformers
sentencepiece
```

3. Run the Streamlit app

```powershell
streamlit run app.py
```

On first run the app may download the fine-tuned model from the Hugging Face Hub (if configured that way).

---

If you'd like, I can also add a short example usage section, a link to the model card on Hugging Face, or a small badge showing build/deploy status.