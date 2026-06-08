"""Cross-modal multi-hop reachability audit for the metabolite->disease
discovery target (re-quantifies docs/kg_inventory_and_discovery_gap.md
sections 6.3 / 6.5 / 6.6 against the *current* final_kg_edges.tsv).

Motivation: after the 825-over-merge bug fix + produces growth (2026-05-28
KG rebuild 3.41M -> 3.61M), the previously reported reachability numbers
(treats_disease micro-path 2.8% / gene-path 12.1%, mediator pool 549) are
stale. The original audit was a throwaway script; this is the durable rerun.

Two reasoning paths for a target edge (metabolite m -> disease d):
  - microbe path:  m <-produces- microbe -enriched/depleted_in-> d
  - gene path:     m -up/downregulates_gene-> host_gene -associated_with_disease-> d

A target is "reachable" via a path iff there exists a shared intermediate
(a microbe producing m that is also enriched/depleted in d; or a gene that m
regulates that is also associated with d).

NOTE on head_type filtering: enriched_in / depleted_in / associated_with_disease
each appear with BOTH microbe and metabolite (or host_gene and metabolite) heads.
Intermediate edges are taken ONLY from the correct head_type slice -- otherwise
metabolite->disease edges would leak into the "microbe path" and inflate coverage.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_EDGES = REPO / "kg_build" / "final_kg_edges.tsv"
DEFAULT_OUT = REPO / "kg_build" / "discovery_audit"

MIC_DIS_RELS = {"enriched_in", "depleted_in"}            # head=microbe -> disease
MET_GENE_RELS = {"upregulates_gene", "downregulates_gene"}  # head=metabolite -> host_gene
GENE_DIS_REL = "associated_with_disease"                 # head=host_gene -> disease
TREATS_REL = "treats_disease"                            # head=metabolite -> disease (target)
ASSOC_REL = "associated_with_disease"                    # head=metabolite -> disease (target)

WANTED_RELS = {"produces", TREATS_REL} | MIC_DIS_RELS | MET_GENE_RELS | {GENE_DIS_REL}


def load_indices(edges_path, chunksize):
    """Single streaming pass over the edge table; build only what the audit needs."""
    produces_by = defaultdict(set)   # metabolite -> {microbe that produces it}
    dis_microbes = defaultdict(set)  # disease    -> {microbe enriched/depleted in it}
    met_genes = defaultdict(set)     # metabolite -> {host_gene it regulates}
    dis_genes = defaultdict(set)     # disease    -> {host_gene associated with it}
    produce_microbes = set()         # microbes with >=1 produces edge
    enriched_microbes = set()        # microbes with >=1 mic->disease edge
    treats_targets = set()           # {(metabolite, disease)} from treats_disease
    assoc_targets = set()            # {(metabolite, disease)} from assoc_with_disease(met)

    reader = pd.read_csv(
        edges_path, sep="\t", dtype=str,
        usecols=["head_id", "head_type", "relation", "tail_id", "tail_type"],
        chunksize=chunksize,
    )
    for chunk in reader:
        sub = chunk[chunk.relation.isin(WANTED_RELS)]
        for h, ht, r, t, tt in zip(sub.head_id, sub.head_type, sub.relation,
                                   sub.tail_id, sub.tail_type):
            if r == "produces" and ht == "microbe" and tt == "metabolite":
                produces_by[t].add(h)
                produce_microbes.add(h)
            elif r in MIC_DIS_RELS and ht == "microbe" and tt == "disease":
                dis_microbes[t].add(h)
                enriched_microbes.add(h)
            elif r in MET_GENE_RELS and ht == "metabolite" and tt == "host_gene":
                met_genes[h].add(t)
            elif r == GENE_DIS_REL and ht == "host_gene" and tt == "disease":
                dis_genes[t].add(h)
            elif r == TREATS_REL and ht == "metabolite" and tt == "disease":
                treats_targets.add((h, t))
            elif r == ASSOC_REL and ht == "metabolite" and tt == "disease":
                assoc_targets.add((h, t))

    return dict(
        produces_by=produces_by, dis_microbes=dis_microbes,
        met_genes=met_genes, dis_genes=dis_genes,
        produce_microbes=produce_microbes, enriched_microbes=enriched_microbes,
        treats_targets=treats_targets, assoc_targets=assoc_targets,
    )


def reachability(targets, idx):
    """Per-target-edge path existence. Returns counts + no-path sample list."""
    micro = gene = either = 0
    no_path = []
    for m, d in targets:
        mp = bool(idx["produces_by"].get(m) and idx["dis_microbes"].get(d)
                  and idx["produces_by"][m] & idx["dis_microbes"][d])
        gp = bool(idx["met_genes"].get(m) and idx["dis_genes"].get(d)
                  and idx["met_genes"][m] & idx["dis_genes"][d])
        micro += mp
        gene += gp
        if mp or gp:
            either += 1
        else:
            no_path.append((m, d))
    n = len(targets)
    return dict(n=n, micro=micro, gene=gene, either=either,
                none=n - either, no_path=no_path)


def marginal(targets, idx):
    """Section 6.5: marginal endpoint coverage over unique metabolites / diseases."""
    mets = {m for m, _ in targets}
    diss = {d for _, d in targets}
    return dict(
        n_met=len(mets), n_dis=len(diss),
        met_micro=sum(1 for m in mets if idx["produces_by"].get(m)),
        met_gene=sum(1 for m in mets if idx["met_genes"].get(m)),
        dis_micro=sum(1 for d in diss if idx["dis_microbes"].get(d)),
        dis_gene=sum(1 for d in diss if idx["dis_genes"].get(d)),
    )


def microbe_gene_bridge(idx):
    """Is the microbe -> metabolite -> host_gene chain wired up?
    A 'bridge metabolite' is produced by >=1 microbe AND regulates >=1 host_gene,
    so it links the microbe modality to the gene modality through metabolism."""
    m_prod = set(idx["produces_by"])   # metabolites with >=1 producing microbe
    m_reg = set(idx["met_genes"])      # metabolites regulating >=1 host_gene
    bridge = m_prod & m_reg
    mic_via, gene_via = set(), set()
    for m in bridge:
        mic_via |= idx["produces_by"][m]
        gene_via |= idx["met_genes"][m]
    return dict(m_prod=len(m_prod), m_reg=len(m_reg), bridge=len(bridge),
                mic_via=len(mic_via), gene_via=len(gene_via))


def pct(n, d):
    return f"{100 * n / d:.2f}%" if d else "-"


def audit(edges_path, out_dir, chunksize, n_samples):
    edges_path = Path(edges_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {edges_path.relative_to(REPO)} (chunksize={chunksize:,}) ...")
    idx = load_indices(edges_path, chunksize)

    mediator = idx["produce_microbes"] & idx["enriched_microbes"]
    print(f"  produces microbes:            {len(idx['produce_microbes']):,}")
    print(f"  mic->disease microbes:        {len(idx['enriched_microbes']):,}")
    print(f"  mediator pool (intersection): {len(mediator):,}")

    treats = reachability(sorted(idx["treats_targets"]), idx)
    assoc = reachability(sorted(idx["assoc_targets"]), idx)
    treats_m = marginal(idx["treats_targets"], idx)
    bridge = microbe_gene_bridge(idx)
    print(f"\nmicrobe<->gene bridge via metabolites: "
          f"{bridge['bridge']:,} bridge metabolites "
          f"({bridge['mic_via']:,} microbes <-> {bridge['gene_via']:,} genes)")

    for name, r in (("treats_disease", treats), ("assoc_with_disease(met)", assoc)):
        print(f"\n{name}: {r['n']:,} unique (met,dis) pairs")
        print(f"  microbe path: {r['micro']:,} ({pct(r['micro'], r['n'])})")
        print(f"  gene    path: {r['gene']:,} ({pct(r['gene'], r['n'])})")
        print(f"  either  path: {r['either']:,} ({pct(r['either'], r['n'])})")
        print(f"  NO path:      {r['none']:,} ({pct(r['none'], r['n'])})")

    # --- samples of unreachable target edges ---
    def save(name, rows):
        if rows:
            pd.DataFrame(rows[:n_samples], columns=["metabolite_id", "disease_id"]).to_csv(
                out_dir / f"no_path_{name}.tsv", sep="\t", index=False)

    save("treats", treats["no_path"])
    save("assoc", assoc["no_path"])

    # --- markdown report ---
    lines = [
        "# Cross-modal reachability audit (discovery feasibility)",
        "",
        f"- edges: `{edges_path.relative_to(REPO)}`",
        "- target = metabolite->disease edges with real curated gold "
        "(`treats_disease`, `associated_with_disease`).",
        "- baseline for comparison: v3.2.1 (3.41M KG) in "
        "`docs/kg_inventory_and_discovery_gap.md` sections 6.3 / 6.5 / 6.6.",
        "",
        "## Mediator microbe pool (section 6.6)",
        "",
        "| set | count |",
        "|---|---:|",
        f"| microbes with a `produces` edge | {len(idx['produce_microbes']):,} |",
        f"| microbes with a `enriched/depleted_in`->disease edge | {len(idx['enriched_microbes']):,} |",
        f"| **mediator pool** (intersection, microbe-path hinge) | **{len(mediator):,}** |",
        "",
        "> v3.2.1 baseline: produces 1,439 / mic-dis 8,685 / mediator **549**.",
        "",
        "## 6.3 Per-edge reachability",
        "",
        "| target | unique (met,dis) | microbe path | gene path | either | **no path** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in (("`treats_disease`", treats), ("`assoc_with_disease`(met)", assoc)):
        lines.append(
            f"| {name} | {r['n']:,} | {r['micro']:,} ({pct(r['micro'], r['n'])}) | "
            f"{r['gene']:,} ({pct(r['gene'], r['n'])}) | "
            f"{r['either']:,} ({pct(r['either'], r['n'])}) | "
            f"**{r['none']:,} ({pct(r['none'], r['n'])})** |")
    lines += [
        "",
        "> v3.2.1 baseline: treats_disease micro 2.8% / gene 12.1% / either 12.7% / "
        "no-path 87.3% (n=6,132); assoc micro 3.4% / gene 9.3% / either 11.0% / "
        "no-path 89.0% (n=12,930).",
        "",
        "## 6.5 Marginal endpoint coverage (treats_disease)",
        "",
        "| path | metabolite end | disease end |",
        "|---|---:|---:|",
        f"| microbe path | {treats_m['met_micro']}/{treats_m['n_met']} "
        f"({pct(treats_m['met_micro'], treats_m['n_met'])}) | "
        f"{treats_m['dis_micro']}/{treats_m['n_dis']} "
        f"({pct(treats_m['dis_micro'], treats_m['n_dis'])}) |",
        f"| gene path | {treats_m['met_gene']}/{treats_m['n_met']} "
        f"({pct(treats_m['met_gene'], treats_m['n_met'])}) | "
        f"{treats_m['dis_gene']}/{treats_m['n_dis']} "
        f"({pct(treats_m['dis_gene'], treats_m['n_dis'])}) |",
        "",
        "> v3.2.1 baseline: met-end micro 18.0% / gene 15.1%; "
        "dis-end micro 16.9% / gene 70.6% (556 met, 1,422 dis).",
        "",
        "## microbe <-> host_gene bridge via metabolites",
        "",
        "Chain: `microbe -produces-> metabolite -up/downregulates_gene-> host_gene`. "
        "A bridge metabolite is produced by a microbe AND regulates a gene.",
        "",
        "| set | count |",
        "|---|---:|",
        f"| metabolites produced by >=1 microbe | {bridge['m_prod']:,} |",
        f"| metabolites regulating >=1 host_gene | {bridge['m_reg']:,} |",
        f"| **bridge metabolites** (both) | **{bridge['bridge']:,}** |",
        f"| microbes reaching >=1 gene via a bridge metabolite | {bridge['mic_via']:,} |",
        f"| host genes reached from >=1 microbe via a bridge | {bridge['gene_via']:,} |",
        "",
        "## Notes",
        "",
        "- Reachability is computed over *unique* (metabolite, disease) pairs, not "
        "raw edge rows (duplicate pairs from multi-source merge share the same path).",
        "- Intermediate edges are head_type-filtered: microbe-path uses only "
        "`enriched/depleted_in` with `head_type=microbe`; gene-path uses only "
        "`associated_with_disease` with `head_type=host_gene`.",
        "- A reachable target is NOT a leakage concern by itself -- it means the KG "
        "*can* offer a structural signal. The no-path fraction is the hard ceiling: "
        "those targets have no 2-hop mediator and a structure-only model cannot learn them.",
        f"- Unreachable target samples (first {n_samples}) in `no_path_*.tsv`.",
    ]
    report = out_dir / "reachability_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {report.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=str(DEFAULT_EDGES))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--n_samples", type=int, default=30)
    args = ap.parse_args()
    audit(args.edges, args.out_dir, args.chunksize, args.n_samples)


if __name__ == "__main__":
    main()
