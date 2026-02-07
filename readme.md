# 📰 News Popularity Intelligence System

A Transformer-based Deep Learning system that predicts the **popularity potential of news articles** using only textual content. Since real engagement labels (clicks, shares, impressions) are unavailable at publish time, the system applies **self-supervised learning and weak supervision** to infer popularity.

---

## 📌 Project Overview

Digital news platforms must decide which articles to highlight before real popularity signals are available. This project reframes popularity as a **latent variable** and infers it indirectly using semantic understanding and linguistic cues.

The system uses:
- Pretrained Transformer models (DistilBERT)
- Proxy popularity signals
- Self-supervised scoring logic
- Explainable AI techniques
- Streamlit-based user interface

---

## 🎯 Objectives

- Learn deep semantic representations of news articles
- Infer popularity without labeled data
- Rank articles based on attention potential
- Provide interpretable explanations
- Build an end-to-end AI application

---

## 🧠 Key Concepts Covered

- Self-Supervised Learning
- Weak Supervision
- Transfer Learning
- Transformer Architecture
- Contextual Embeddings
- Ranking Systems
- Feature Engineering
- Explainable AI (XAI)
- NLP Sentiment & Readability Analysis
- Modular Pipeline Design

---

## 📂 Dataset Information

- Type: Unlabeled News Text Dataset
- Columns Used:
  - Title
  - Description
- Nature: Real-world editorial content
- Labels: Not available

Only raw text is used in accordance with project constraints.

---

## 🏗️ Project Structure

```
news_popularity_intelligence/
│
├── data/
│   ├── News_dataset.csv
│   ├── news_embeddings.pt
│   ├── proxy_signals.csv
│   └── ranked_news.csv
│
├── notebooks/
│   ├── 00_debug_dataset.py
│   ├── 01_data_loading.py
│   ├── 02_text_preprocessing.py
│   ├── 03_transformer_embeddings.py
│   ├── 04_proxy_signal_design.py
│   └── 05_popularity_scoring.py
│
├── app/
│   └── streamlit_app.py
│
├── utils/
│   ├── text_utils.py
│   ├── proxy_signals.py
│   └── scoring.py
│
└── README.md
```

---

## ⚙️ System Architecture

```
Input Text (Title + Description)
        ↓
Text Construction
        ↓
Transformer Encoder (DistilBERT)
        ↓
CLS Embeddings (768-d)
        ↓
Proxy Signal Extraction
        ↓
Self-Supervised Scoring
        ↓
Popularity Ranking
        ↓
Streamlit Interface
```

---

## 🚀 Implementation Steps

### Step 1: Data Loading
- Load CSV file
- Validate columns
- Perform sanity checks

File: `01_data_loading.py`

---

### Step 2: Text Preprocessing
- Handle missing values
- Merge Title and Description
- Add separator token `[SEP]`

File: `02_text_preprocessing.py`

---

### Step 3: Transformer Embeddings
- Load DistilBERT model
- Tokenize text
- Extract CLS embeddings
- Save embeddings

File: `03_transformer_embeddings.py`

---

### Step 4: Proxy Signal Design

Weak supervision signals:
- Emotional Intensity
- Subjectivity
- Urgency Score
- Lexical Diversity
- Readability Score
- Length Balance

All signals are normalized.

File: `04_proxy_signal_design.py`

---

### Step 5: Popularity Scoring & Ranking

- Weighted aggregation of proxy signals
- Compute pseudo-popularity score
- Rank articles
- Save ranked output

File: `05_popularity_scoring.py`

---

### Step 6: Streamlit Application

Pages:
1. Home Page
2. News Intelligence Page
3. Model Reasoning Page

File: `app/streamlit_app.py`

---

## 📊 Evaluation Methodology

Since labeled popularity data is unavailable:

- Qualitative Evaluation
- Ranking Consistency Analysis
- Case Studies
- Human-Interpretable Explanations

Traditional accuracy metrics are not used.

---

## 🔒 Project Constraints

- Transformer-based models only
- No traditional ML algorithms
- No manual labeling
- No supervised popularity datasets
- CPU-friendly execution

All constraints are strictly followed.

---

## 💻 Installation & Setup

### 1. Install Dependencies

```bash
pip install torch transformers textblob tqdm streamlit pandas numpy
```

---

### 2. Run Pipeline Scripts

```bash
python notebooks/01_data_loading.py
python notebooks/02_text_preprocessing.py
python notebooks/03_transformer_embeddings.py
python notebooks/04_proxy_signal_design.py
python notebooks/05_popularity_scoring.py
```

---

### 3. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Output Files

| File | Description |
|------|-------------|
| news_embeddings.pt | Transformer embeddings |
| proxy_signals.csv | Weak supervision signals |
| ranked_news.csv | Ranked popularity output |

---

## 🧪 Sample Results

- Embedding Dimension: 768
- Sample Size: 2000 (CPU-safe subset)
- Output: Ranked news articles with explainations

---

## 🧠 Design Justification

- DistilBERT chosen for CPU efficiency
- CLS token used for global representation
- Proxy signals replace missing labels
- Ranking preferred over classification
- Subsampling ensures system stability

---

## 📜 Limitations

- No real engagement data
- Heuristic-based supervision
- Limited to textual features
- CPU-based scalability constraints

---

## 🔮 Future Enhancements

- GPU-based full dataset training
- Pairwise ranking models
- Fine-tuning Transformer models
- Multilingual support
- Real-time API deployment

---

## 👨‍💻 Author

Name: Vijay Ragavan S

---

## 📄 License

This project is developed for academic and research purposes.

---

## ✅ Conclusion

This project demonstrates a complete Transformer-based News Popularity Intelligence System using self-supervised learning and explainable AI principles. It aligns with industry practices and academic evaluation standards.

