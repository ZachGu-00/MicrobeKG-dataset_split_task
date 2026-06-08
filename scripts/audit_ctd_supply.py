"""Ceiling analysis for Plan-2 (CTD chem-gene expansion).

Question: the gene path is `metabolite -regulates-> host_gene -assoc-> disease`.
Its metabolite end is the bottleneck (14.4% of treats gold metabolites regulate
any gene). The current import (step_ctd_chem_gene.py Plan B) only kept CTD
chem-gene edges whose metabolite is microbe-linked AND action is expression.
If we re-extract with a gold-linked filter instead, how high can we push:
  (a) the metabolite-end coverage, and
  (b) the END-TO-END gene-path reachability (the number that actually matters)?

Two tiers of relaxation:
  - expr      : keep only increases/decreases expression (current schema, just
                drop the microbe-linked filter)
  - all-action: any CTD interaction (binding/activity/metabolic/...) -- upper
                bound; would need a broader regulates_gene relation.

Reuses load_indices() (audit_reachability) and the CTD xref mapping
(step_ctd_chem_gene) so numbers are identical to production extraction.
"""
from collections import defaultdict
from pathlib import Path

from audit_reachability import load_indices, DEFAULT_EDGES, pct
from step_ctd_chem_gene import (
    load_kg_metabolites, build_chem_to_kgid, CTD_IXN_FILE, ACTION_TO_RELATION,
)

OUT = Path(__file__).resolve().parents[1] / "kg_build" / "discovery_audit"


def scan_ctd(chem_to_kgid):
    """metabolite_id -> set(HGNC gene) from CTD chem-gene (Homo sapiens), 2 tiers."""
    expr = defaultdict(set)    # expression actions only (current schema)
    allact = defaultdict(set)  # any interaction action
    n_human = 0
    with CTD_IXN_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 11 or p[6] != "Homo sapiens":
                continue
            n_human += 1
            kg_mid = chem_to_kgid.get(p[1])
            if not kg_mid:
                continue
            gene = "HGNC:" + p[3]
            acts = [a.strip() for a in p[9].split("|") if a.strip()]
            allact[kg_mid].add(gene)
            if any(a in ACTION_TO_RELATION for a in acts):
                expr[kg_mid].add(gene)
    return expr, allact, n_human


def met_end(G, met_genes, extra):
    """metabolites in G that regulate >=1 gene, now vs now+extra."""
    now = sum(1 for m in G if met_genes.get(m))
    plus = sum(1 for m in G if met_genes.get(m) or extra.get(m))
    return now, plus


def genepath(targets, met_genes, dis_genes, extra):
    """end-to-end: exists gene regulated by m AND associated with d."""
    c = 0
    for m, d in targets:
        mg = met_genes.get(m, set()) | extra.get(m, set())
        dg = dis_genes.get(d)
        if mg and dg and (mg & dg):
            c += 1
    return c


def block(label, targets, idx, expr, allact):
    G = {m for m, _ in targets}
    n = len(targets)
    mg, dg = idx["met_genes"], idx["dis_genes"]
    me_now, me_expr = met_end(G, mg, expr)
    _, me_all = met_end(G, mg, allact)
    gp_now = genepath(targets, mg, dg, {})
    gp_expr = genepath(targets, mg, dg, expr)
    gp_all = genepath(targets, mg, dg, allact)
    lines = [
        "### " + label + "  (" + format(n, ",") + " unique pairs, "
        + format(len(G), ",") + " gold metabolites)",
        "",
        "| | metabolite-end coverage | gene-path reachability |",
        "|---|---:|---:|",
        "| current (Plan B) | " + str(me_now) + "/" + str(len(G)) + " (" + pct(me_now, len(G))
        + ") | " + str(gp_now) + "/" + str(n) + " (" + pct(gp_now, n) + ") |",
        "| + CTD expr (drop microbe-linked) | " + str(me_expr) + "/" + str(len(G)) + " ("
        + pct(me_expr, len(G)) + ") | " + str(gp_expr) + "/" + str(n) + " (" + pct(gp_expr, n) + ") |",
        "| + CTD any-action (upper bound) | " + str(me_all) + "/" + str(len(G)) + " ("
        + pct(me_all, len(G)) + ") | " + str(gp_all) + "/" + str(n) + " (" + pct(gp_all, n) + ") |",
        "",
    ]
    print("\n".join(lines))
    return lines


def main():
    print("Loading KG indices ...")
    idx = load_indices(DEFAULT_EDGES, 500_000)
    treats = sorted(idx["treats_targets"])
    assoc = sorted(idx["assoc_targets"])

    print("Building CTD chemical -> KG metabolite map ...")
    _, *byx = load_kg_metabolites()
    chem_to_kgid = build_chem_to_kgid(*byx)
    print("  CTD chemicals mappable to KG metabolite: " + format(len(chem_to_kgid), ","))

    print("Scanning CTD chem-gene (588MB) ...")
    expr, allact, n_human = scan_ctd(chem_to_kgid)
    print("  human ixns scanned: " + format(n_human, ","))
    print("  mappable metabolites with >=1 gene edge: expr=" + format(len(expr), ",")
          + "  any-action=" + format(len(allact), ",") + "\n")

    body = []
    body += block("treats_disease", treats, idx, expr, allact)
    body += block("assoc_with_disease(met)", assoc, idx, expr, allact)

    OUT.mkdir(parents=True, exist_ok=True)
    header = [
        "# Plan-2 ceiling: CTD chem-gene expansion (gold-linked re-extract)",
        "",
        "- CTD chemicals mappable to KG metabolite (via ctd_chemical_xref.tsv): "
        + format(len(chem_to_kgid), ","),
        "- CTD human ixns scanned: " + format(n_human, ",")
        + "; mappable metabolites with a gene edge: expr " + format(len(expr), ",")
        + " / any-action " + format(len(allact), ","),
        "- metabolite-end = gold metabolites regulating >=1 gene; "
        "gene-path reachability = end-to-end metabolite->gene->disease with a shared gene.",
        "- gene-path disease end (70.7% of treats diseases have a gene assoc) is NOT "
        "the bottleneck, so metabolite-end gains transfer to reachability.",
        "",
    ]
    (OUT / "ctd_supply_ceiling.md").write_text("\n".join(header + body), encoding="utf-8")
    print("Report -> kg_build/discovery_audit/ctd_supply_ceiling.md")


if __name__ == "__main__":
    main()
