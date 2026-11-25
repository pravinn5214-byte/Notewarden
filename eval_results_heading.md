# Retrieval evaluation results

Generated: 2026-11-25 07:03 · 45 questions · extractor: PyMuPDF + clean_text · chunking: heading · child overlap 20 · RRF k=60 · reranker: ms-marco-MiniLM-L-6-v2 · 1000 bootstrap resamples (seed 42)
## Mode comparison — child size 120

Values: mean [bootstrap 95% CI]. CI overlap ⇒ not significant.

| mode | k | overall hit/MRR | keyword | semantic |
|---|---|---|---|---|
| dense | 3 | 0.863 [0.77-0.95] | 0.885 [0.77-0.98] | 0.833 [0.68-0.96] |
| dense | 5 | 0.873 [0.78-0.95] | 0.894 [0.79-0.98] | 0.844 [0.70-0.96] |
| hybrid | 3 | 0.863 [0.76-0.94] | 0.885 [0.77-0.98] | 0.833 [0.67-0.96] |
| hybrid | 5 | 0.874 [0.78-0.95] | 0.904 [0.82-0.98] | 0.833 [0.67-0.96] |
| hybrid_rerank | 3 | 0.911 [0.84-0.97] | 0.936 [0.86-1.00] | 0.877 [0.76-0.97] |
| hybrid_rerank | 5 | 0.911 [0.84-0.97] | 0.936 [0.86-1.00] | 0.877 [0.76-0.97] |
