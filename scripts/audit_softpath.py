"""Soft-path / leverage analysis for the microbe path (responds to the point
that 1.27% hard-path coverage is an explicit-metapath artifact, not a KGC
ceiling). All numbers computed against the current final_kg_edges.tsv.

(B1) Soft path: drop the 'same microbe' intersection. Count treats/assoc pairs
     where SOME microbe produces m AND SOME (possibly different) microbe is DA
     in d. Tests whether coverage rebounds toward the disease-end ceiling.
(B2) AGORA2 lever: of the bottleneck microbes (in microbe->disease edges but
     with NO produces edge), how many are in AGORA2 (can_utilize) and thus have
     a GEM to FBA-infer produces edges.
(B3) Genus rollup: project produces / DA microbes to genus via belongs_to_genus
     and recompute the mediator pool (strain-level vs genus-level).
(C)  metabolite->disease direct edges already in the KG (the metabolite-side
     bypass that lets a KGE model triangulate without the microbe mediator).
"""
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EDGES = REPO / "kg_build" / "final_kg_edges.tsv"
OUT = REPO / "kg_build" / "discovery_audit"
MIC_DIS = {"enriched_in", "depleted_in"}


def load(path, chunksize=500_000):
    produces_by = defaultdict(set)   # metabolite -> {producing microbe}
    dis_microbes = defaultdict(set)  # disease    -> {DA microbe}
    produce_mic, dis_mic, canutil_mic = set(), set(), set()
    genus_of = {}                    # microbe -> genus node
    treats, assoc = set(), set()
    met_dis = Counter()              # metabolite->disease relation counts
    reader = pd.read_csv(path, sep="\t", dtype=str,
                         usecols=["head_id", "head_type", "relation", "tail_id", "tail_type"],
                         chunksize=chunksize)
    for ch in reader:
        for h, ht, r, t, tt in zip(ch.head_id, ch.head_type, ch.relation, ch.tail_id, ch.tail_type):
            if ht == "microbe" and tt == "metabolite" and r == "produces":
                produces_by[t].add(h); produce_mic.add(h)
            elif ht == "microbe" and tt == "disease" and r in MIC_DIS:
                dis_microbes[t].add(h); dis_mic.add(h)
            elif ht == "microbe" and r == "can_utilize":
                canutil_mic.add(h)
            elif ht == "microbe" and tt == "microbe" and r == "belongs_to_genus":
                genus_of[h] = t
            elif ht == "metabolite" and tt == "disease":
                met_dis[r] += 1
                if r == "treats_disease":
                    treats.add((h, t))
                elif r == "associated_with_disease":
                    assoc.add((h, t))
    return dict(produces_by=produces_by, dis_microbes=dis_microbes,
                produce_mic=produce_mic, dis_mic=dis_mic, canutil_mic=canutil_mic,
                genus_of=genus_of, treats=treats, assoc=assoc, met_dis=met_dis)


def pct(n, d):
    return format(100 * n / d, ".2f") + "%" if d else "-"


def coverage(targets, produces_by, dis_microbes):
    n = len(targets)
    hard = soft = me = de = 0
    for m, d in targets:
        sm = produces_by.get(m)
        sd = dis_microbes.get(d)
        if sm:
            me += 1
        if sd:
            de += 1
        if sm and sd:
            soft += 1
            if sm & sd:
                hard += 1
    return dict(n=n, hard=hard, soft=soft, met_end=me, dis_end=de)


def main():
    print("Loading KG (streaming) ...")
    d = load(EDGES)
    produces_by, dis_microbes = d["produces_by"], d["dis_microbes"]

    lines = ["# Microbe-path leverage analysis (soft path / AGORA2 / genus / bypass)", ""]

    # B1 soft path
    lines += ["## B1. Soft path (drop the same-microbe intersection)", "",
              "| target | n | hard (same microbe) | soft (any microbe both ends) | met-end | dis-end |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, tg in (("treats_disease", d["treats"]), ("assoc_with_disease", d["assoc"])):
        c = coverage(sorted(tg), produces_by, dis_microbes)
        lines.append("| " + name + " | " + str(c["n"]) + " | "
                     + str(c["hard"]) + " (" + pct(c["hard"], c["n"]) + ") | "
                     + str(c["soft"]) + " (" + pct(c["soft"], c["n"]) + ") | "
                     + pct(c["met_end"], c["n"]) + " | " + pct(c["dis_end"], c["n"]) + " |")
        print("B1 " + name + ": hard " + pct(c["hard"], c["n"]) + " -> soft "
              + pct(c["soft"], c["n"]) + " (met-end " + pct(c["met_end"], c["n"])
              + ", dis-end " + pct(c["dis_end"], c["n"]) + ")")

    # B2 AGORA2 lever
    bottleneck = d["dis_mic"] - d["produce_mic"]
    b_agora = bottleneck & d["canutil_mic"]
    lines += ["", "## B2. AGORA2 lever (FBA-infer produces for bottleneck microbes)", "",
              "| set | count |", "|---|---:|",
              "| microbes with a produces edge | " + str(len(d["produce_mic"])) + " |",
              "| microbes in microbe->disease edges | " + str(len(d["dis_mic"])) + " |",
              "| bottleneck (disease-linked, NO produces) | " + str(len(bottleneck)) + " |",
              "| ...of which in AGORA2 (can_utilize, has GEM) | " + str(len(b_agora))
              + " (" + pct(len(b_agora), len(bottleneck)) + ") |",
              "| AGORA2 microbe pool (can_utilize) | " + str(len(d["canutil_mic"])) + " |"]
    print("B2 bottleneck " + str(len(bottleneck)) + ", in AGORA2 " + str(len(b_agora))
          + " (" + pct(len(b_agora), len(bottleneck)) + ")")

    # B3 genus rollup
    g = d["genus_of"]
    prod_g = {g.get(m, m) for m in d["produce_mic"]}
    dis_g = {g.get(m, m) for m in d["dis_mic"]}
    med_strain = d["produce_mic"] & d["dis_mic"]
    med_genus = prod_g & dis_g
    lines += ["", "## B3. Genus rollup of the mediator pool", "",
              "| level | mediator pool (produces microbes that are also DA microbes) |",
              "|---|---:|",
              "| strain (current) | " + str(len(med_strain)) + " |",
              "| genus rollup | " + str(len(med_genus)) + " |",
              "", "(" + str(len(g)) + " microbes carry a belongs_to_genus mapping)"]
    print("B3 mediator: strain " + str(len(med_strain)) + " -> genus " + str(len(med_genus)))

    # C metabolite->disease bypass
    lines += ["", "## C. metabolite->disease direct edges (KGE bypass)", "",
              "| relation | count |", "|---|---:|"]
    for r, c in d["met_dis"].most_common():
        lines.append("| " + r + " | " + str(c) + " |")
    print("C metabolite->disease: " + str(dict(d["met_dis"])))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "softpath_leverage.md").write_text("\n".join(lines), encoding="utf-8")
    print("Report -> kg_build/discovery_audit/softpath_leverage.md")


if __name__ == "__main__":
    main()
