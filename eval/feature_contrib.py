import json
import numpy as np

def load_cp(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def stats(rows):
    return {
        "r5":         round(float(np.mean([r["r5"]  for r in rows])), 4),
        "r10":        round(float(np.mean([r["r10"] for r in rows])), 4),
        "mrr":        round(float(np.mean([r["mrr"] for r in rows])), 4),
        "faith":      round(float(np.mean([r["faith"] for r in rows])), 4),
        "ret_lat":    round(float(np.mean([r["ret_lat"] for r in rows])), 3),
        "rerank_lat": round(float(np.mean([r.get("rerank_lat", 0) for r in rows])), 3),
        "total_lat":  round(float(np.mean([r["total_lat"] for r in rows])), 3),
    }

base  = stats(load_cp("eval/data/checkpoints/eval2_Vector_Only.jsonl"))
hybrid = stats(load_cp("eval/data/checkpoints/eval2_Hybrid_VectorplusBM25.jsonl"))
rrf   = stats(load_cp("eval/data/checkpoints/eval2_HybridplusRRF.jsonl"))
prod  = stats(load_cp("eval/data/checkpoints/eval2_Production_Full.jsonl"))
rec   = stats(load_cp("eval/data/checkpoints/eval1_recursive.jsonl"))
sem   = stats(load_cp("eval/data/checkpoints/eval1_semantic.jsonl"))

def pct(new, old):
    if abs(old) < 1e-9:
        return 0.0
    return round((new - old) / old * 100, 1)

def delta(new, old):
    return round(new - old, 4)

print("=" * 62)
print("FEATURE CONTRIBUTION BREAKDOWN")
print("=" * 62)

print()
print("STEP 0 — Naive baseline (vector-only + semantic chunking)")
print(f"  Recall@5  = {base['r5']:.4f}")
print(f"  Recall@10 = {base['r10']:.4f}")
print(f"  MRR       = {base['mrr']:.4f}")
print(f"  Retrieval latency = {base['ret_lat']}s")

print()
print("STEP 1 — Add BM25 hybrid (no RRF)")
print(f"  Recall@5  = {hybrid['r5']:.4f}  ({delta(hybrid['r5'],  base['r5']):+.4f} / {pct(hybrid['r5'],  base['r5']):+.1f}%)")
print(f"  Recall@10 = {hybrid['r10']:.4f}  ({delta(hybrid['r10'], base['r10']):+.4f} / {pct(hybrid['r10'], base['r10']):+.1f}%)")
print(f"  MRR       = {hybrid['mrr']:.4f}  ({delta(hybrid['mrr'], base['mrr']):+.4f} / {pct(hybrid['mrr'], base['mrr']):+.1f}%)")
print(f"  Note: BM25 alone without fusion = no gain in this experiment")

print()
print("STEP 2 — Add RRF fusion (Hybrid + RRF, still no reranker)")
print(f"  Recall@5  = {rrf['r5']:.4f}  ({delta(rrf['r5'],  base['r5']):+.4f} / {pct(rrf['r5'],  base['r5']):+.1f}%) vs base")
print(f"  Recall@10 = {rrf['r10']:.4f}  ({delta(rrf['r10'], base['r10']):+.4f} / {pct(rrf['r10'], base['r10']):+.1f}%) vs base")
print(f"  MRR       = {rrf['mrr']:.4f}  ({delta(rrf['mrr'], base['mrr']):+.4f} / {pct(rrf['mrr'], base['mrr']):+.1f}%) vs base")
print(f"  RRF alone adds: R@5 {delta(rrf['r5'], hybrid['r5']):+.4f}  MRR {delta(rrf['mrr'], hybrid['mrr']):+.4f}")

print()
print("STEP 3 — Add CrossEncoder reranker (Full Production)")
print(f"  Recall@5  = {prod['r5']:.4f}  ({delta(prod['r5'],  base['r5']):+.4f} / {pct(prod['r5'],  base['r5']):+.1f}%) vs base")
print(f"  Recall@10 = {prod['r10']:.4f}  ({delta(prod['r10'], base['r10']):+.4f} / {pct(prod['r10'], base['r10']):+.1f}%) vs base")
print(f"  MRR       = {prod['mrr']:.4f}  ({delta(prod['mrr'], base['mrr']):+.4f} / {pct(prod['mrr'], base['mrr']):+.1f}%) vs base")
print(f"  Reranker adds: R@5 {delta(prod['r5'], rrf['r5']):+.4f}  R@10 {delta(prod['r10'], rrf['r10']):+.4f}  MRR {delta(prod['mrr'], rrf['mrr']):+.4f}")
print(f"  Reranker latency cost: +{prod['rerank_lat']}s per query")

print()
print("=" * 62)
print("CHUNKING STRATEGY CONTRIBUTION (both w/ full pipeline)")
print("=" * 62)
print(f"  Recursive: R@5={rec['r5']}  R@10={rec['r10']}  MRR={rec['mrr']}  latency={rec['total_lat']}s")
print(f"  Semantic:  R@5={sem['r5']}  R@10={sem['r10']}  MRR={sem['mrr']}  latency={sem['total_lat']}s")
print(f"  Semantic gain: R@5 {delta(sem['r5'], rec['r5']):+.4f} ({pct(sem['r5'], rec['r5']):+.1f}%)  MRR {delta(sem['mrr'], rec['mrr']):+.4f} ({pct(sem['mrr'], rec['mrr']):+.1f}%)")

print()
print("=" * 62)
print("TOTAL LIFT: Naive baseline → Full Production System")
print("=" * 62)
# Estimated naive baseline: recursive chunking + vector-only
# Scale recursive_full down by the same ratio vector_only/production on semantic
rec_vec_est = round(rec["r5"] * (base["r5"] / prod["r5"]), 4)
print(f"  Estimated naive baseline (recursive + vector-only): R@5 ≈ {rec_vec_est}")
print(f"  Full system (semantic + hybrid + RRF + reranker):   R@5 = {prod['r5']}")
total_lift = round(prod["r5"] - rec_vec_est, 4)
total_pct  = round(total_lift / rec_vec_est * 100, 1)
print(f"  Total absolute gain:  +{total_lift} Recall@5")
print(f"  Total relative gain:  +{total_pct}%")
print()
print("  Feature breakdown of total gain:")
rrf_contrib  = delta(rrf["r5"],  base["r5"])   # RRF+Hybrid vs vector-only
rank_contrib = delta(prod["r5"], rrf["r5"])     # reranker on top of RRF
chunk_contrib = delta(sem["r5"], rec["r5"])      # semantic vs recursive (full pipeline)
vec_vs_base  = total_lift - rrf_contrib - rank_contrib - chunk_contrib
print(f"    Semantic chunking over recursive: +{chunk_contrib:.4f} R@5")
print(f"    RRF fusion (BM25+RRF vs vector):  +{rrf_contrib:.4f} R@5")
print(f"    CrossEncoder reranker:            +{rank_contrib:.4f} R@5")
print(f"    (residual / estimation error):    +{round(vec_vs_base,4):.4f} R@5")
