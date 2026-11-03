#!/usr/bin/env python3
"""Build the data part of the licence-aware MicrobeKG-Bench v3 release.

Inputs are read-only; only --out is written.  Graph triples and splits are stored as Parquet with stable entity and
relation identifiers (never server-local row numbers).  Triples supported only by CTD/HMDB are excluded; the
full-context setting is represented by a rebuild recipe.  Validation and test tables carry a `study_id` column (the
evidence cluster used for study-macro MRR and the hierarchical bootstrap: first literature study of the pair, else its
cohort group; for the external set, the BugSigDB study).  `finalize_release.py` writes the manifest and checksums.

Example:
  python make_release.py --bench bench2 --bench3 bench3 --external bench_ext --records records_v2.tsv \
      --edges final_kg_edges.tsv --source-terms-json source_terms.json --source-terms-md SOURCE_TERMS.md --out out
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

RESTRICTED = {"CTD", "HMDB"}
SEEDS = [42, 123, 456, 789, 2024]
SETTINGS = ["t1_pair", "t1_study", "t1_study_reveal", "t1_study_strict", "t1_study_ctx_open", "t1_temporal",
            "t1_temporal_nodis", "t1_cold_microbe", "t1_cold_disease"]
V3 = ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc",
      "t1_v3_temporal_dated", "t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other"]
PARTS = ["train", "valid", "test", "known_extra", "conflict", "negatives", "probe_positive", "random_control_positive"]
COLS = ["head_id", "head_type", "relation", "tail_id", "tail_type"]


def write_json(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + "\n")


def triples_df(x, ents, rels):
    names, types = ents.node_id.to_numpy(), ents.node_type.to_numpy()
    return pd.DataFrame({"head_id": names[x[:, 0]], "head_type": types[x[:, 0]],
                         "relation": np.asarray(rels, dtype=object)[x[:, 1]], "tail_id": names[x[:, 2]],
                         "tail_type": types[x[:, 2]]})


def write_split(raw, x, keep_codes, ents, rels, sd, seed, study_of):
    n = len(ents)
    for part in PARTS:
        vals = raw[part] if part in raw.files else np.array([], np.int32)
        rows = vals if vals.ndim == 2 else np.asarray(x[vals])
        rows = (np.asarray([r for r in rows if tuple(map(int, r)) in keep_codes], np.int32).reshape((-1, 3))
                if len(rows) else np.empty((0, 3), np.int32))
        df = triples_df(rows, ents, rels) if len(rows) else pd.DataFrame(columns=COLS)
        if part in ("valid", "test"):
            code = rows[:, 0].astype(np.int64) * n + rows[:, 2]
            df["study_id"] = [study_of.get(int(c), "") for c in code]
        df.to_parquet(sd / f"seed_{seed}_{part}.parquet", index=False)
    elig = raw["candidate_eligible"]
    pd.DataFrame({"entity_index": np.flatnonzero(elig), "node_id": ents.node_id.to_numpy()[elig],
                  "node_type": ents.node_type.to_numpy()[elig]}).to_parquet(
        sd / f"seed_{seed}_candidate_entities.parquet", index=False)
    np.savez_compressed(sd / f"seed_{seed}_aux.npz", **{k: raw[k] for k in raw.files if k.endswith("_eligible") or
                                                         k in ("taxonomy_hop", "ontology_hop", "random_control")})


def main():
    ap = argparse.ArgumentParser()
    for a in ["bench", "bench3", "external", "records", "edges", "source-terms-json", "source-terms-md", "out"]:
        ap.add_argument(f"--{a}", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new output directory")
    out.mkdir(parents=True)
    b, b3, e = args.bench, args.bench3, args.external
    ents = pd.read_csv(b / "entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    rels = json.loads((b / "relations.json").read_text())
    x = np.load(b / "triples.npy")
    n, R = len(ents), len(rels)
    idx = {(t, i): j for j, t, i in ents[["entity_index", "node_type", "node_id"]].itertuples(index=False, name=None)}
    rix = {r: i for i, r in enumerate(rels)}
    code = lambda h, r, t: (h.astype(np.int64) * R + r) * n + t
    xcode = code(x[:, 0], x[:, 1], x[:, 2])
    assert np.all(np.diff(xcode) > 0), "server triples must be sorted and unique"
    restricted = np.ones(len(x), bool)
    for df in pd.read_csv(args.edges, sep="\t", usecols=COLS + ["source"], dtype=str, keep_default_na=False,
                          chunksize=500000):
        h = np.fromiter((idx[(a, c)] for a, c in zip(df.head_type, df.head_id)), np.int64, len(df))
        t = np.fromiter((idx[(a, c)] for a, c in zip(df.tail_type, df.tail_id)), np.int64, len(df))
        r = np.fromiter((rix[v] for v in df.relation), np.int64, len(df))
        pos = np.searchsorted(xcode, code(h, r, t))
        ok = np.fromiter((bool(set(s.split("|")) - RESTRICTED) for s in df.source), bool, len(df))
        restricted[pos[ok]] = False
    keep = ~restricted
    removed = int(restricted.sum())
    bd = out / "benchmark"
    (bd / "splits").mkdir(parents=True)
    ents.to_csv(bd / "entities.tsv", sep="\t", index=False)
    write_json(bd / "relations.json", rels)
    triples_df(x[keep], ents, rels).to_parquet(bd / "triples.parquet", index=False)
    pub = set(map(tuple, x[keep].tolist()))
    ps = pd.read_csv(b / "pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
    ps.to_parquet(bd / "pair_support.parquet", index=False)
    l1, c1 = ps.lit_studies.str.split(";").str[0], ps.cohort_studies.str.split(";").str[0]
    study_of = dict(zip(ps.code.astype(np.int64).tolist(), np.where(l1 != "", l1, c1).tolist()))
    pd.read_csv(b / "tested_null.tsv", sep="\t", dtype=str, keep_default_na=False).to_parquet(
        out / "tested_null.parquet", index=False)
    for setting in SETTINGS + V3:
        src_root = b3 if setting in V3 else b
        sd = bd / "splits" / setting
        sd.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            src = src_root / "splits" / setting / f"seed_{seed}"
            meta = json.loads(src.with_suffix(".json").read_text())
            raw = np.load(src.with_suffix(".npz"), allow_pickle=False)
            write_split(raw, x, pub, ents, rels, sd, seed, study_of)
            write_json(sd / f"seed_{seed}.json", {**meta, "released_representation": "triple_parquet",
                                                   "source_row_indices_not_released": True})
        print(setting, "done", flush=True)
    # P1-1 pair sets (revealed same-study pairs, their matched subset, and the matched other-study reserve)
    pd_dir = bd / "p11_pairs"
    pd_dir.mkdir()
    names = ents.node_id.to_numpy()
    for seed in SEEDS:
        z = np.load(b3 / f"p11_pairs_seed{seed}.npz")
        frames = [pd.DataFrame({"set": k, "head_id": names[z[k] // n], "tail_id": names[z[k] % n]})
                  for k in ["revealed", "matched", "reserve"]]
        pd.concat(frames).to_parquet(pd_dir / f"seed_{seed}.parquet", index=False)
    write_json(bd / "variants_build_audit.json", json.loads((b3 / "build_audit.json").read_text()))
    write_json(bd / "t1_study_ctx_full_REBUILD_REQUIRED.json", {
        "setting": "t1_study_ctx_full", "status": "rebuild-required",
        "reason": "training context includes triples supported only by CTD/HMDB; source terms restrict redistribution",
        "script": "mkbench_build_v2.py in the code supplement",
        "inputs": "the audited graph and source permissions; not included"})
    # external time-out set
    ext_x = np.load(e / "triples.npy", mmap_mode="r")
    ext_codes = set(map(tuple, ext_x.tolist()))
    exd = out / "external"
    exd.mkdir()
    shutil.copy2(e / "entities.tsv", exd / "entities.tsv")
    shutil.copy2(e / "relations.json", exd / "relations.json")
    ext = pd.read_csv(e / "external_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
    ext["study_id"] = ext.bsdb.str.split("/").str[0]                 # BugSigDB study (PMID- or DOI-keyed)
    ext.loc[ext.study_id.str.startswith("bsdb:10."), "pmid"] = ""     # DOI-keyed studies have no PMID
    ext.to_parquet(exd / "external_pairs.parquet", index=False)
    ext_study = ext.groupby(ext.code.astype(np.int64)).study_id.first().to_dict()
    ents_e = pd.read_csv(e / "entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    rels_e = json.loads((e / "relations.json").read_text())
    sd = exd / "splits" / "t1_external"
    sd.mkdir(parents=True)
    for seed in SEEDS:
        src = e / "splits" / "t1_external" / f"seed_{seed}"
        raw = np.load(src.with_suffix(".npz"), allow_pickle=False)
        write_split(raw, ext_x, ext_codes, ents_e, rels_e, sd, seed, ext_study)
        write_json(sd / f"seed_{seed}.json", {**json.loads(src.with_suffix(".json").read_text()),
                                               "released_representation": "triple_parquet",
                                               "source_row_indices_not_released": True})
    # record layer without personal-information columns
    rec = pd.read_csv(args.records, sep="\t", dtype=str, keep_default_na=False)
    pii = [c for c in rec.columns if c.lower() in {"authors", "author", "curator", "submittedby", "revision_editor",
                                                   "email", "name", "location_of_subjects"}]
    rec = rec.drop(columns=pii, errors="ignore")
    (out / "records").mkdir()
    rec.to_parquet(out / "records" / "records_v2.parquet", index=False)
    write_json(out / "records" / "schema.json", {"columns": {c: {"dtype": "string", "personal_information": False,
                                                                 "source": "derived upstream record export"}
                                                             for c in rec.columns}, "removed_columns": pii})
    write_json(out / "SOURCE_TERMS.json", json.loads(args.source_terms_json.read_text()))
    shutil.copy2(args.source_terms_md, out / "SOURCE_TERMS.md")
    write_json(out / "build_info.json", {"ctd_hmdb_only_triples_removed": removed, "settings": SETTINGS + V3,
                                         "seeds": SEEDS, "graph_version": "microbekg-audited",
                                         "benchmark_version": "mkbench-v3"})
    for root, dirs, files in os.walk(out):
        for d in dirs:
            os.chmod(Path(root) / d, 0o755)
        for f in files:
            os.chmod(Path(root) / f, 0o644)
    print(json.dumps({"out": str(out), "removed_ctd_hmdb_only": removed}), flush=True)


if __name__ == "__main__":
    main()
