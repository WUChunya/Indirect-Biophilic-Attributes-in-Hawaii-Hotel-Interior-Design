"""Command line: `biophilic <stage>` (or `python -m biophilic <stage>`)."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config, pipeline


def _load_state() -> dict:
    p = config.path("processed")
    if not (p / "reviews.parquet").exists():
        return pipeline.ingest()
    reviews = pd.read_parquet(p / "reviews.parquet")
    return {"reviews": reviews, "booking_hotels": pd.read_parquet(p / "booking_hotels.parquet"),
            "coding": pd.read_parquet(p / "bdia_coding.parquet"), "bdia_hotels": pd.read_parquet(p / "bdia_hotels.parquet"),
            **json.loads((p / "ingest_meta.json").read_text())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="biophilic", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="read raw Excel files → data/processed/")
    sub.add_parser("baseline", help="ingest + lexicon + BDIA indices + models → results/REPORT.md")
    s = sub.add_parser("annotate-sample", help="write the gold-set annotation sheet → data/interim/")
    s.add_argument("--n", type=int, default=None)
    sub.add_parser("absa-submit", help="submit all text reviews to the Claude Message Batches API")
    sub.add_parser("absa-collect", help="collect batch results → data/processed/mentions_llm.parquet")
    sub.add_parser("absa-sample", help="run the LLM extractor on the annotation sample (synchronous)")
    sub.add_parser("evaluate-text", help="score lexicon and LLM against the annotated gold set")
    sub.add_parser("vision-code", help="code BDIA items from photos in data/external/photos/")
    sub.add_parser("vision-agreement", help="VLM coding vs 2024 human coding → results/vision_agreement.csv")
    args = ap.parse_args(argv)

    interim, processed, results = config.path("interim"), config.path("processed"), config.path("results")
    interim.mkdir(parents=True, exist_ok=True)

    if args.cmd == "ingest":
        st = pipeline.ingest()
        print(f"{len(st['reviews'])} reviews, {len(st['bdia_hotels'])} BDIA hotels → {processed}")
    elif args.cmd == "baseline":
        st = pipeline.ingest()
        mentions = pipeline.text_baseline(st["reviews"], st["booking_hotels"])
        res = pipeline.analyse(st, mentions)
        pipeline.write_results(res)
        print(f"wrote {results / 'REPORT.md'}")
    elif args.cmd == "annotate-sample":
        from .text import annotation

        st = _load_state()
        mentions = pd.read_parquet(processed / "mentions_lexicon.parquet")
        aspects = list(config.load("review_aspects")["aspects"])
        n = args.n or config.load("pipeline")["text"]["annotation_sample_size"]
        df = annotation.sample(st["reviews"], mentions, aspects, n, config.load("pipeline")["text"]["random_seed"])
        path = interim / "annotation_sample.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}: fill one column per aspect with +, - or 0")
    elif args.cmd == "absa-submit":
        from .text import llm_absa

        st = _load_state()
        ids = llm_absa.submit(st["reviews"], interim / "llm_batches.json")
        print(f"submitted {len(ids)} batch(es): {ids}")
    elif args.cmd == "absa-collect":
        from .text import llm_absa

        st = _load_state()
        df = llm_absa.collect(interim / "llm_batches.json", st["reviews"])
        df.to_parquet(processed / "mentions_llm.parquet")
        print(f"{len(df)} mentions; {len(df.attrs['failed'])} requests failed (see attrs)")
    elif args.cmd == "absa-sample":
        from .text import llm_absa

        sample = pd.read_csv(interim / "annotation_sample.csv")
        df = llm_absa.run_sync(sample)
        df.to_parquet(interim / "mentions_llm_sample.parquet")
        print(f"{len(df)} mentions on {len(sample)} reviews; failed: {df.attrs['failed']}")
    elif args.cmd == "evaluate-text":
        from .text import annotation

        gold = pd.read_csv(interim / "annotation_sample.csv").fillna("")
        aspects = list(config.load("review_aspects")["aspects"])
        frames = []
        lex = pd.read_parquet(processed / "mentions_lexicon.parquet")
        frames.append(annotation.evaluate(lex[lex["method"] == "lexicon"], gold, aspects).assign(method="lexicon"))
        frames.append(annotation.evaluate(lex, gold, aspects).assign(method="lexicon+ambiguous (2024-like)"))
        llm_path = interim / "mentions_llm_sample.parquet"
        if llm_path.exists():
            frames.append(annotation.evaluate(pd.read_parquet(llm_path), gold, aspects).assign(method="llm"))
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(results / "text_evaluation.csv", index=False)
        print(out.groupby("method")[["precision", "recall", "f1"]].mean().round(3))
    elif args.cmd == "vision-code":
        from .vision import vlm_coder

        df = vlm_coder.code_all(config.path("photos"), processed / "bdia_vlm_long.parquet")
        print(f"{df[['hotel_id', 'space']].drop_duplicates().shape[0] if len(df) else 0} hotel × space coded")
    elif args.cmd == "vision-agreement":
        from . import agreement, bdia
        from .vision import vlm_coder

        st = _load_state()
        thr = config.load("pipeline")["llm"]["vision_presence_threshold"]
        vlm = vlm_coder.to_wide(pd.read_parquet(processed / "bdia_vlm_long.parquet"), thr)
        tab = agreement.agreement_table(st["coding"], vlm, bdia.item_codes())
        tab.to_csv(results / "vision_agreement.csv", index=False)
        print(tab[["percent_agreement", "cohen_kappa", "gwet_ac1"]].mean().round(3))


if __name__ == "__main__":
    main()
