# 🧠 AI Data Engineering Suite — IT Platform Edition

> **Role-Based Platform · Self-Healing Pipeline · Data Quality AI · Pipeline Monitoring ·
> Power BI-style Analytics · Security & RBAC · Reasoning LLM Agent · Vector Search**
> Built with Python · Streamlit · Three.js · Plotly · scikit-learn · XGBoost · ChromaDB · LangGraph · OpenRouter

A professional, IT-company-style **Data Engineering & Data Analytics platform** with
role-based dashboards (Data Engineer / Data Analyst / Admin), a security layer, and
pipeline monitoring.

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Add your API key (for AI features)
Copy `.env.example` to **`.env`** and set your key — it is **never hardcoded**:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```
Get a free key at https://openrouter.ai/keys. (The app still runs without a key — AI
buttons just show a friendly "add your key" message.)

### 3. Run
```bash
streamlit run app.py      # or: run.bat  (Windows one-click)
```
Open http://localhost:8501

### 4. Log in (demo accounts)
| Username | Password | Role | Sees |
|----------|----------|------|------|
| `admin` | `admin123` | **Admin** | everything + Security center |
| `engineer` | `engineer123` | **Data Engineer** | pipelines, monitoring, quality |
| `analyst` | `analyst123` | **Data Analyst** | dashboards, KPIs, forecasting |

> 🔐 **Security:** the API key lives only in `.env`/secrets, passwords are SHA-256 hashed,
> PII (email/phone/salary/customer-id) is masked for non-admin roles, and every action is
> written to the audit log.

---

## ✨ What's New (Pro · Agentic Edition)

| Feature | Description |
|---------|-------------|
| 🕸️ **Agentic pipeline** | A **LangGraph** state machine autonomously runs profile → detect → heal → quality → ML → AI-summary, with a live execution trace (sequential fallback if LangGraph is absent). |
| 🧬 **Vector database** | **ChromaDB** (cosine HNSW) indexes rows via OpenRouter embeddings; powers Semantic Search & RAG. Auto-falls back to a numpy cosine store. |
| 💬 **RAG "Ask your data"** | Retrieval-augmented Q&A — top-k rows pulled from the vector DB and used as grounded evidence (with citations). |
| 📈 **ML analytics** | **XGBoost** predictive modeling (auto target + feature importance) and **IsolationForest** anomaly detection. |
| 🔒 **Hidden API key** | Key embedded in `config.py`, resolved via secrets → env → embedded. Never shown in the UI. |
| 🧠 **Reasoning models** | Default `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` + `openai/gpt-oss-120b:free`; chat & agent show a **reasoning-trace** expander. |
| 🎨 **Pro UI** | Space Grotesk + Inter typography, glassy gradient theme, pill-style sidebar nav, gauge + radar charts, unified Plotly theme. |
| 🚀 **Pro landing** | Full-screen hero (sidebar hidden) with a **Launch** button and animated 3D field. |

> 🛟 Every heavy dependency (ChromaDB, XGBoost, LangGraph) is **optional** — the app
> auto-detects and gracefully falls back, so it always runs end-to-end.

---

## 🗂️ Pages

| Page | What It Does |
|------|-------------|
| 🏠 Landing | Marketing site: top navbar (Home/Features/About/Contact), "Big Data" 3D sphere, Launch button |
| 📂 Data Hub | Upload your CSV (replaces sidebar data source); preview + jump-to-tool shortcuts |
| 🛠️ Data Engineer View | Role dashboard: pipeline status, ETL, data quality, error logs, concepts |
| 📡 Pipeline Monitoring | Success/failure, freshness, records, errors, processing time, alerts |
| 🏗️ Data Engineering Guide | Ingestion · ETL/ELT · cleaning · transformation · warehouse · automation |
| 📊 Data Analyst View | Power BI-style: KPIs, sales trend, regional, product, customer, forecast |
| 🛡️ Admin / Security | Users & RBAC, PII masking, audit log, backups, encryption explainer |
| 📚 IT Data Roles Guide | DE vs DA, tools, security, career path, real companies |
| 📊 Dashboard | 3D pipeline viz, distributions, correlation matrix, before/after |
| 🕸️ Agentic Pipeline | One-click autonomous LangGraph workflow + AI summary |
| 🔧 Self-Healing Pipeline | Detect → Fix → Alert + **✨ one-click AI fix & corrected-CSV download** |
| 🔍 Data Quality AI | 6-dimension scoring + **gauge & radar** + SQL/code fixes |
| 📈 AI Analytics | XGBoost modeling · IsolationForest anomalies · RAG Q&A |
| 🤖 AI Agent Chat | DataSage multi-turn chat **with reasoning traces** |
| 🔎 Semantic Search | Vector-DB meaning-based row search |
| 📑 Executive Report | AI-generated, downloadable stakeholder report |
| 📁 Data Explorer | Filter, search, profile any column |

---

## 🧱 Tech Stack
Streamlit · Plotly · Three.js · pandas/numpy/scipy · **scikit-learn** · **XGBoost** ·
**ChromaDB** · **LangGraph / LangChain** · OpenRouter (reasoning LLMs + embeddings)

---

## 🔑 Models (free-first, via OpenRouter)

| Model | Use | Cost |
|-------|-----|------|
| 🧠 Nemotron Nano Omni Reasoning | chat (default) | Free |
| ⚡ GPT-OSS 120B | chat | Free |
| 🦙 Llama 3.1 8B / 🔥 Mistral 7B | chat | Free |
| 🧬 Llama-Nemotron Embed VL | embeddings | Free |
| 💎 Gemini Flash · 🧪 Claude 3.5 | chat | Paid |

---

## 🔧 Self-Healing Pipeline
| Detection | Auto-Fix |
|-----------|----------|
| Missing / NULL values | Median / mode imputation |
| Outliers (IQR 3x) | Capping to statistical bounds |
| Type errors (text in numeric) | Coerce + median fill |
| Negative anomalies | Replace with positive median |
| Duplicate rows | Auto-deduplication |
| Whitespace (new) | Auto-trim |
| Constant columns (new) | Flagged as low-information |

## 🔍 Data Quality AI (6 Dimensions)
Completeness · Accuracy · Consistency · Uniqueness · Timeliness · Validity — each
scored, explained, and paired with SQL/Python fix code.

---

## 📁 Project Structure
```
ai_data_pipeline/
├── app.py                    # Main Streamlit app (role-based, 16 pages)
├── config.py                 # API key from .env / secrets (NOT hardcoded)
├── run.bat                   # One-click Windows launcher
├── requirements.txt
├── .env.example / .gitignore
├── .streamlit/
│   ├── config.toml           # Dark theme defaults
│   └── secrets.toml.example  # Optional key override
├── modules/
│   ├── self_healing.py       # Detectors + auto-fixers + health score
│   ├── data_quality.py       # 6-dimension quality scanner
│   ├── ai_agent.py           # OpenRouter agent: chat, reasoning, embeddings, RAG, insights
│   ├── vector_store.py       # ChromaDB vector index (+ numpy fallback)
│   ├── ml_analytics.py       # XGBoost predictive + IsolationForest anomalies
│   ├── agent_graph.py        # LangGraph agentic orchestration
│   ├── security.py           # Login, RBAC, PII masking, audit, encryption explainer
│   ├── monitoring.py         # Pipeline metrics, freshness, alerts
│   └── analytics.py          # Sales trend, regional, product, customer, forecasting
├── data/                     # Walmart + Navneet datasets + navneet_export,
│                             #   employee_access_log, pipeline_audit_log
└── components/               # Three.js 3D visuals (incl. big-data sphere)
```

---

## 🧪 Sample Datasets (intentional issues for demo)
Walmart Sales · Walmart Inventory · Walmart Customers · Navneet Products ·
Navneet Sales · Navneet Financials — each seeded with realistic data-quality issues.

---

## 👤 Author
**Snehal Laxman Jadhav** — AI Engineer at **Navneet Education Limited** · © 2026

## 📜 License
MIT — free for personal and commercial use.

> ⚠️ The embedded key is shared with anyone who receives these files. Rotate it at
> [openrouter.ai/keys](https://openrouter.ai/keys) if it leaks.
