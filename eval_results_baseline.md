# Retrieval evaluation results

Generated: 2026-11-25 05:21 · 45 questions · extractor: PyMuPDF + clean_text · child overlap 20 · RRF k=60 · reranker: ms-marco-MiniLM-L-6-v2 · 1000 bootstrap resamples (seed 42)
## Mode comparison — child size 120

Values: mean [bootstrap 95% CI]. CI overlap ⇒ not significant.

| mode | k | overall hit/MRR | keyword | semantic |
|---|---|---|---|---|
| dense | 3 | 0.667 [0.55-0.78] | 0.699 [0.56-0.83] | 0.623 [0.43-0.81] |
| dense | 5 | 0.681 [0.57-0.79] | 0.706 [0.57-0.83] | 0.646 [0.47-0.81] |
| hybrid | 3 | 0.752 [0.64-0.86] | 0.788 [0.67-0.90] | 0.702 [0.49-0.87] |
| hybrid | 5 | 0.763 [0.66-0.86] | 0.798 [0.70-0.90] | 0.715 [0.53-0.87] |
| hybrid_rerank | 3 | 0.878 [0.79-0.94] | 0.904 [0.81-0.98] | 0.842 [0.71-0.95] |
| hybrid_rerank | 5 | 0.888 [0.81-0.95] | 0.913 [0.83-0.98] | 0.853 [0.74-0.95] |
