# Mise — AI Demand Forecasting for Independent Restaurants

> *"A weekly AI demand forecast for independent restaurants — telling them what to order before they waste money finding out the hard way."*

Final project for **2758-T4 Advanced Topics in Machine Learning**  
Nova School of Business and Economics · Masters in Business Analytics · 2026

---

## The Problem

Independent restaurants run on 3–5% margins and make inventory decisions by gut feel. A 40-seat bistro wastes **€600–2,000/month** in spoiled food — not because owners are careless, but because they have no tool that tells them what demand will look like next week.

## The Solution

Mise connects to a restaurant's POS system (Square, Lightspeed), analyses sales history alongside weather and local event signals, and delivers a plain-English Monday-morning ordering brief:

> *"Order 15% less salmon — rain forecast Friday–Sunday. Order 20% more pasta — Festas de Lisboa this Saturday."*

One brief. One decision. Less waste.

## Repository Structure

```
├── BUSINESS_PLAN.md          # Deliverable 1 — Business plan, GTM, financials, GenAI log
├── data/                     # Synthetic dataset (generated Week 1)
│   └── da_mario_sales.csv
├── models/                   # ML components
│   ├── forecasting/          # LightGBM demand forecasting model
│   ├── cold_start/           # Bayesian cold-start module
│   └── anomaly/              # Isolation Forest anomaly detection
├── backend/                  # FastAPI backend + Claude API integration
│   └── main.py
├── frontend/                 # Streamlit dashboard
│   └── app.py
└── notebooks/                # EDA, model training, evaluation
```

## ML Architecture

| Component | Method | Purpose |
|---|---|---|
| Demand forecasting | LightGBM | Predict next-week item demand |
| Cold-start | Bayesian clustering | Credible prior for new restaurants |
| Anomaly detection | Isolation Forest | Flag broken signals |
| Recommendation text | Claude API | Plain-English weekly brief |

## Team

| Person | Role |
|---|---|
| Person 1 | Business plan, GTM strategy, financial model, pitch |
| Person 2 | Synthetic data, LightGBM model, Bayesian cold-start, anomaly detection |
| Person 3 | FastAPI backend, POS simulation, Claude API integration |
| Person 4 | Streamlit frontend, dashboard UI, demo flow |

## GenAI Transparency

All AI tool usage is fully documented in **[Appendix A of the Business Plan](BUSINESS_PLAN.md#appendix-a--genai-transparency-log)**, in compliance with the course's academic integrity requirements.

---

*Nova SBE · 2026*
