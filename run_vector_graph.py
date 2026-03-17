import argparse
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core_vector_graph import run_pure, run_graph

def _fmt_int(v):
    return str(v) if v is not None else "N/A"

def _fmt_float(v, nd=4):
    if v is None: return "N/A"
    try: return f"{float(v):.{nd}f}"
    except Exception: return "N/A"

def print_top_hits(top_hits):
    print("\n=== TOP HITS ===")
    for tv in top_hits:
        print(f"#{tv.rank} score={tv.score:.4f} pid:{tv.pid} src:{tv.src} dst:{tv.dst} text:{tv.text}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--mode", choices=["pure", "graph"], required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--topk", type=int, default=400)
    ap.add_argument("--join_topk", type=int, default=2000)
    ap.add_argument("--max_evidence", type=int, default=20)
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--model_embed", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    client = QdrantClient(url=args.qdrant_url)
    embedder = SentenceTransformer(args.model_embed, local_files_only=args.offline)

    if args.mode == "pure":
        out = run_pure(client, embedder, args.collection, args.q, args.topk, args.max_evidence)
    else:
        out = run_graph(client, embedder, args.collection, args.q, args.topk, args.join_topk, args.max_evidence)

    print("\n=== UNIFIED METRICS ===")
    print(f"q={out['q']}")
    print(f"mode={out['mode']}")
    print(f"topk={_fmt_int(out['topk'])}")
    print(f"join_topk={_fmt_int(out['join_topk'])}")
    print(f"Kmax={_fmt_int(out['k_max'])}")
    print(f"Kshown={_fmt_int(out['k_used'])}")
    print(f"retrieved_hits={_fmt_int(out['retrieved_hits'])}")
    print(f"candidates_len={_fmt_int(out['candidates_len'])}")
    print(f"candidates_unique_src={_fmt_int(out['candidates_unique_src'])}")
    print(f"pid_eval={out['pid_eval'] or 'N/A'}")
    print(f"RawSuccess@K={_fmt_int(out.get('raw_success_at_k'))}")
    print(f"RawRankFirstValid={_fmt_int(out.get('raw_rank_first_valid'))}")
    print(f"RawValid@K={_fmt_int(out.get('raw_valid_at_k'))}")
    print(f"RawNoise@K={_fmt_int(out.get('raw_noise_at_k'))}")
    print(f"Success@K={_fmt_int(out['success_at_k'])}")
    print(f"RankFirstValid={_fmt_int(out['rank_first_valid'])}")
    print(f"Valid@K={_fmt_int(out['valid_at_k'])}")
    print(f"Noise@K={_fmt_int(out['noise_at_k'])}")
    print(f"num_clauses={_fmt_int(out['num_clauses'])}")
    print(f"hits_raw_total={_fmt_int(out['hits_raw_total'])}")
    print(f"hits_meta_total={_fmt_int(out['hits_meta_total'])}")
    print(f"unique_src_meta_total={_fmt_int(out['unique_src_meta_total'])}")
    print(f"join_success={_fmt_int(out['join_success'])}")
    print(f"join_common_src={_fmt_int(out['join_common_src'])}")
    print(f"join_rate={_fmt_float(out['join_rate'])}")
    print(f"join_answers_shown={_fmt_int(out['join_answers_shown'])}")
    print(f"first_joined_src={out['first_joined_src'] or 'N/A'}")

    if out.get('top_hits'):
        print_top_hits(out['top_hits'])
    elif out.get('joined_ranked'):
        print(f"\n=== JOINED SRC (top {args.max_evidence}) ===")
        for i, row in enumerate(out["joined_ranked"], 1):
            print(f"#{i} src={row.get('src')} sum_score={row.get('sum_score', 0.0):.4f}")

    if out.get('first_valid'):
        fv = out['first_valid']
        print("\n=== FIRST VALID HIT ===")
        print(f"rank={fv.rank} | src={fv.src} ({fv.src_label}) | dst={fv.dst} ({fv.dst_label}) | year={fv.matched_year or 'N/A'}")
        print(f"text:{fv.text}")

if __name__ == "__main__":
    main()