#

Evaluation cases: 13
Query rewrites: frozen
Embedding: BAAI/bge-small-zh-v1.5
BM25: k1=1.5, b=0.75
Hybrid: RRF k=60
Multi-query: original + 2 rewrites
Reranker: BAAI/bge-reranker-base
Candidate count: 30
Final retrieval window: 8

MultiQuery+Hybrid

- Candidate Complete@30: 1.000
- Complete@8: 0.769
- Recall@8: ~0.808
- Top1: 0.308
- MRR: ~0.47

MultiQuery+Hybrid+Reranker

- Candidate Complete@30: 1.000
- Complete@8: 0.769
- Recall@8: 0.769
- Complete@5: 0.692
- Recall@5: 0.731
- Top1: 0.231
- MRR: ~0.46-0.47  
- 