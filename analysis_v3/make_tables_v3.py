"""Write every LaTeX table and every number macro of the paper from the analysis CSVs (no hand-copied numbers).

Inputs: results3/analysis (main analyses), results2/analysis and results2/records (decomposition, closure, cold sides,
record layer), audit_hmdad (classic protocol audit, KGNMDA), release/v3 (source terms, split tables).
Outputs: manuscript_v3/tables/*.tex and manuscript_v3/tables/numbers.tex (\\newcommand macros used in the text).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/user/ClawsGO/MicrobeKG")
A3, A2, REC = R / "results3/analysis", R / "results2/analysis", R / "results2/records"
OUT = R / "manuscript_v3/tables"
OUT.mkdir(parents=True, exist_ok=True)

NAME = {"popularity": "Degree prior", "cn": "CN", "ra": "RA", "l3": "L3", "transe": "TransE", "distmult": "DistMult",
        "complex": "ComplEx", "rotate": "RotatE", "tucker": "TuckER", "rgcn": "R-GCN", "nbfnet": "NBFNet",
        "ncn": "NCN", "ultra-zeroshot": "ULTRA (zero-shot)", "ultra-finetune": "ULTRA (fine-tuned)",
        "gbm": "GBM (pair features)", "random": "Random", "rotate-rs": "RotatE (random search)",
        "tucker-rs": "TuckER (random search)", "rgcn-rs": "R-GCN (random search)"}
HEUR = ["popularity", "cn", "ra", "l3"]
EMB = ["transe", "distmult", "complex", "rotate", "tucker"]
MP = ["rgcn", "nbfnet", "ncn", "ultra-zeroshot", "ultra-finetune"]
FEAT = ["gbm"]
RS = ["rotate-rs", "tucker-rs", "rgcn-rs"]
LEARNED = EMB + MP + FEAT                  # label counts and summary statements use these
ROWS = HEUR + EMB + MP + FEAT
ALLLEARNED = LEARNED + RS                  # appendix tables also show the random-search refits
BREAK = {"l3", "tucker", "ultra-finetune", "gbm"}
HEAD = {"t1_pair": "Pair", "t1_study_reveal": "Study+R", "t1_study": "Study-excl.", "t1_study_strict": "Strict",
        "t1_temporal": "Temporal", "t1_external": "Out-of-time", "t1_cold_microbe": "Cold mic.",
        "t1_cold_disease": "Cold dis.", "t1_temporal_nodis": "Temp.\\,$-$Dis.", "t1_study_ctx_open": "Study+ctx",
        "t1_v3_strict_nocohort": "Strict $-$ cohort", "t1_v3_strict_nooverlap": "Strict $-$ BioProject overlap",
        "t1_v3_strict_nocooc": "Strict $-$ co-occurrence", "t1_v3_temporal_nocooc": "Temporal $-$ co-occurrence",
        "t1_v3_temporal_dated": "Temporal-dated"}
TRANSFER = ["t1_study_strict", "t1_study", "t1_temporal", "t1_external"]
LABMARK = {"worse": "$^{\\downarrow}$", "better": "$^{\\uparrow}$", "equivalent": "$^{\\approx}$", "inconclusive": ""}
NUM = {}          # macro name -> text


def num(name, value):
    NUM[name] = value
    return value


MINUS = __import__("re").compile(r"(?<![A-Za-z0-9-])-(?=\d)")


def tex_minus(t):
    """Typeset a leading minus of a number as a math minus (keeps 2-5 ranges, 1e-4 and -- dashes)."""
    return MINUS.sub("$-$", t)


def write(name, body):
    body = tex_minus(body)
    (OUT / name).write_text(body)
    print("wrote", name)


def tab(spec, header, lines, pre=""):
    return (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{pre}{header} \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def f3(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def sgn(v, d=3):
    return "--" if pd.isna(v) else f"{v:+.{d}f}"


def ci(lo, hi, d=3):
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


main = pd.read_csv(A3 / "main_table.csv").set_index(["setting", "model"])
vs = pd.read_csv(A3 / "vs_degree.csv").set_index(["setting", "model"])
present = lambda regs, rows: [m for m in rows if any((s, m) in main.index for s in regs)]

# ------------------------------------------------------------------------------------------ Table 1: regimes, units
sib = pd.read_csv(A3 / "sibling_units.csv").groupby("setting").mean(numeric_only=True)
REGDESC = [("t1_pair", "Random 10\\% of pairs"),
           ("t1_study_reveal", "Study-excl., plus half of each held-out study's pairs in training"),
           ("t1_study", "Pairs reported only by held-out literature studies (10\\% of 1,571)"),
           ("t1_study_strict", "Study-excl., minus every training pair a held-out study co-reports"),
           ("t1_temporal", "Pairs first reported in 2023 or later"),
           ("t1_external", "BugSigDB signatures curated after the cut-off"),
           ("t1_cold_microbe", "All pairs of 10\\% of microbes ($\\geq$5 pairs)"),
           ("t1_cold_disease", "All pairs of 10\\% of diseases ($\\geq$5 pairs)")]
lines = []
for s, desc in REGDESC:
    n_ = f"{main.loc[(s, 'popularity'), 'n_test']:,.0f}"
    name = HEAD[s] + (" (main)" if s == "t1_study_strict" else "")
    if s in sib.index:
        r = sib.loc[s]
        c = " & ".join(f"{100 * r[k]:.1f}" for k in ["sib_literature", "sib_gmrepo", "sib_mbodymap", "sib_any"])
    else:
        c = "0$^\\ast$ & 0$^\\ast$ & 0$^\\ast$ & 0$^\\ast$"
    lines.append(f"{name} & {desc} & {n_} & {c} \\\\")
write("regimes.tex", tab("lp{4.9cm}rrrrr", "Regime & Held out & Test triples & Lit. & GMrepo & mBodyMap & Any", lines,
                         pre=" & & & \\multicolumn{4}{c}{Test pairs with a same-unit sibling (\\%)} \\\\\n"
                             "\\cmidrule(lr){4-7}\n"))
p = sib.loc["t1_pair"]
num("SibPairLit", f"{100 * p.sib_literature:.1f}")
num("SibPairGM", f"{100 * p.sib_gmrepo:.1f}")
num("SibPairMB", f"{100 * p.sib_mbodymap:.1f}")
num("SibPairAny", f"{100 * p.sib_any:.1f}")
num("SibStudyLit", f"{100 * sib.loc['t1_study'].sib_literature:.1f}")
num("SibRevealLit", f"{100 * sib.loc['t1_study_reveal'].sib_literature:.1f}")
num("SibTemporalLit", f"{100 * sib.loc['t1_temporal'].sib_literature:.1f}")
for s, k in [("t1_study_strict", "Strict"), ("t1_temporal", "Temporal"), ("t1_external", "Ext"), ("t1_pair", "Pair")]:
    num(f"NTest{k}", f"{main.loc[(s, 'popularity'), 'n_test']:,.0f}")

# ------------------------------------------------------------------------------------------ Table 2: main MRR
MAINREG = TRANSFER + ["t1_pair", "t1_cold_microbe", "t1_cold_disease"]


def mrr_lines(regs, rows, col="mrr", marks=True):
    best = {s: max((main.loc[(s, m), col] for m in rows if (s, m) in main.index), default=np.nan) for s in regs}
    out = []
    for m in rows:
        cells = []
        for s in regs:
            if (s, m) not in main.index:
                cells.append("--")
                continue
            v = main.loc[(s, m), col]
            t = f"{v:.3f}"
            if np.isclose(v, best[s]):
                t = f"\\textbf{{{t}}}"
            if marks and (s, m) in vs.index:
                t += LABMARK[vs.loc[(s, m), f"{col}_label"]]
            cells.append(t)
        out.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
        if m in BREAK:
            out.append("\\midrule")
    while out and out[-1] == "\\midrule":
        out.pop()
    return out


rows_main = present(MAINREG, ROWS)
write("main_results.tex", tab("l" + "r" * len(MAINREG), "Method & " + " & ".join(HEAD[s] for s in MAINREG),
                              mrr_lines(MAINREG, rows_main),
                              pre="& \\multicolumn{4}{c}{Transfer to new studies or later curation} & "
                                  "\\multicolumn{1}{c}{Reference} & \\multicolumn{2}{c}{New endpoints} \\\\\n"
                                  "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}\n"))
for s, k in [("t1_study_strict", "Strict"), ("t1_study", "Study"), ("t1_temporal", "Temporal"), ("t1_external", "Ext"),
             ("t1_pair", "Pair")]:
    num(f"Prior{k}", f3(main.loc[(s, "popularity"), "mrr"]))
    num(f"Prior{k}CI", ci(main.loc[(s, "popularity"), "mrr_lo"], main.loc[(s, "popularity"), "mrr_hi"]))
num("RotatePair", f3(main.loc[("t1_pair", "rotate"), "mrr"]))

# ------------------------------------------------------------------------------------------ label counts (main text)
lab_rows, n_cmp = [], {}
for met, nm in [("mrr", "MRR"), ("h10", "Hits@10"), ("macro", "Study-macro MRR")]:
    w = vs[vs.index.get_level_values(0).isin(TRANSFER) & vs.index.get_level_values(1).isin(LEARNED)]
    c = w[f"{met}_label"].value_counts()
    mde = w[f"{met}_mde80"]
    n_cmp[met] = len(w)
    lab_rows.append(f"{nm} & {len(w)} & " + " & ".join(str(int(c.get(k, 0))) for k in
                                                      ["better", "equivalent", "inconclusive", "worse"])
                    + f" & {mde.min():.3f}--{mde.max():.3f} \\\\")
    for k in ["better", "equivalent", "inconclusive", "worse"]:
        num(f"Lab{met.capitalize()}{k.capitalize()}", str(int(c.get(k, 0))))
    num(f"Lab{met.capitalize()}N", str(len(w)))
    num(f"Mde{met.capitalize()}Lo", f"{mde.min():.3f}")
    num(f"Mde{met.capitalize()}Hi", f"{mde.max():.3f}")
write("label_counts.tex", tab("lrrrrrr", "Metric & Comparisons & Better & Equivalent & Inconclusive & Worse & MDE$_{80}$",
                              lab_rows))
w = vs[vs.index.get_level_values(0).isin(TRANSFER) & vs.index.get_level_values(1).isin(LEARNED)].reset_index()
num("NLearned", str(w.model.nunique()))
for s, k in [("t1_study_strict", "Strict"), ("t1_external", "Ext")]:
    q = w[w.setting == s].mrr_mde80
    num(f"Mde{k}Lo", f"{q.min():.3f}")
    num(f"Mde{k}Hi", f"{q.max():.3f}")
# named exceptions for the text
for (s, m), r in vs.iterrows():
    if s in TRANSFER and m in LEARNED:
        for met in ["mrr", "h10"]:
            if r[f"{met}_label"] in ("better", "equivalent"):
                key = f"Ex{met.capitalize()}{HEAD[s].replace('.', '').replace('-', '')}" + "".join(
                    ch for ch in NAME[m] if ch.isalpha())
                if r[f"{met}_label"] == "better":
                    num(key, f"{r[f'{met}_delta']:+.4f} {ci(r[f'{met}_lo95'], r[f'{met}_hi95'], 4)}")
                else:
                    num(key, f"{r[f'{met}_delta']:+.3f} (90\\% CI {ci(r[f'{met}_lo90'], r[f'{met}_hi90'])})")
if ("t1_study_strict", "tucker") in vs.index:
    r = vs.loc[("t1_study_strict", "tucker")]
    num("TuckerStrictDelta", f"{r.mrr_delta:+.3f}")
    num("TuckerStrictCI", ci(r.mrr_lo95, r.mrr_hi95))
    num("TuckerStrictNinety", ci(r.mrr_lo90, r.mrr_hi90))

# ------------------------------------------------------------------------------------------ appendix: paired differences
for met, fname, cap in [("mrr", "app_vsdeg_mrr.tex", "MRR"), ("h10", "app_vsdeg_h10.tex", "Hits@10"),
                        ("macro", "app_vsdeg_macro.tex", "macro")]:
    lines = []
    for m in present(TRANSFER, [x for x in ROWS + RS if x != "popularity"]):
        cells = []
        for s in TRANSFER:
            if (s, m) not in vs.index:
                cells.append("--")
                continue
            r = vs.loc[(s, m)]
            t = f"{r[f'{met}_delta']:+.3f} {{\\tiny{ci(r[f'{met}_lo95'], r[f'{met}_hi95'])}}}"
            lab = r[f"{met}_label"]
            cells.append(f"\\textbf{{{t}}}" if lab in ("better", "worse") else (f"\\textit{{{t}}}" if lab == "equivalent" else t))
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
        if m in BREAK:
            lines.append("\\midrule")
    mde = " & ".join(f"{vs.loc[[i for i in vs.index if i[0] == s and i[1] in LEARNED], f'{met}_mde80'].min():.3f}--"
                     f"{vs.loc[[i for i in vs.index if i[0] == s and i[1] in LEARNED], f'{met}_mde80'].max():.3f}"
                     for s in TRANSFER)
    lines.append("\\midrule")
    lines.append(f"MDE$_{{80}}$ (learned) & {mde} \\\\")
    write(fname, tab("l" + "c" * len(TRANSFER), "Method & " + " & ".join(HEAD[s] for s in TRANSFER), lines))

# MRR with intervals, all regimes
for tag, regs in [("a", ["t1_pair", "t1_study_reveal", "t1_study", "t1_study_strict"]),
                  ("b", ["t1_temporal", "t1_temporal_nodis", "t1_external", "t1_cold_microbe", "t1_cold_disease"])]:
    lines = []
    for m in present(regs, ROWS + ["random"]):
        cells = [f"{main.loc[(s, m), 'mrr']:.3f} {{\\tiny{ci(main.loc[(s, m), 'mrr_lo'], main.loc[(s, m), 'mrr_hi'])}}}"
                 if (s, m) in main.index else "--" for s in regs]
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    write(f"app_mrr_ci_{tag}.tex", tab("l" + "c" * len(regs), "Method & " + " & ".join(HEAD[s] for s in regs), lines))
ALLREG = ["t1_pair", "t1_study_reveal", "t1_study", "t1_study_strict", "t1_temporal", "t1_temporal_nodis",
          "t1_external", "t1_cold_microbe", "t1_cold_disease"]
for col, fname in [("h10", "app_hits10.tex"), ("macro", "app_macro.tex")]:
    write(fname, tab("l" + "r" * len(ALLREG), "Method & " + " & ".join(HEAD[s] for s in ALLREG),
                     mrr_lines(ALLREG, present(ALLREG, ROWS), col=col, marks=False)))

# ------------------------------------------------------------------------------------------ robustness: variants
vp = pd.read_csv(A3 / "variant_vs_parent.csv")
lines = []
for v in ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc",
          "t1_v3_temporal_dated"]:
    q = vp[vp.variant == v].set_index("model")
    if not len(q):
        continue
    pr = q.loc["popularity"] if "popularity" in q.index else None
    lrn = q[q.index.isin(LEARNED)].dropna(subset=["gap_change"])
    gaps = (f"{lrn.gap_change.min():+.3f} to {lrn.gap_change.max():+.3f} ({len(lrn)})" if len(lrn) else "--")
    neg = int((lrn.gap_hi95 < 0).sum()) if len(lrn) else 0
    pos = int((lrn.gap_lo95 > 0).sum()) if len(lrn) else 0
    sv = vs.loc[[i for i in vs.index if i[0] == v and i[1] in LEARNED]] if v in vs.index.get_level_values(0) else None
    labs = ("--" if sv is None or not len(sv) else
            f"{int((sv.mrr_label == 'better').sum())}/{int((sv.mrr_label == 'worse').sum())}/{len(sv)}")
    lines.append(f"{HEAD[v]} & {sgn(pr.delta) if pr is not None else '--'} {{\\tiny{ci(pr.lo95, pr.hi95) if pr is not None else ''}}} & "
                 f"{gaps} & {neg} / {pos} & {labs} \\\\")
    key = {"t1_v3_strict_nocohort": "Nocohort", "t1_v3_strict_nooverlap": "Nooverlap", "t1_v3_strict_nocooc": "Nocooc",
           "t1_v3_temporal_nocooc": "TNocooc", "t1_v3_temporal_dated": "Dated"}[v]
    if pr is not None:
        num(f"Var{key}Prior", sgn(pr.delta))
        num(f"Var{key}PriorCI", ci(pr.lo95, pr.hi95))
    if len(lrn):
        num(f"Var{key}GapNeg", str(neg))
        num(f"Var{key}GapN", str(len(lrn)))
        num(f"Var{key}GapLo", f"{lrn.gap_change.min():+.3f}")
        num(f"Var{key}GapHi", f"{lrn.gap_change.max():+.3f}")
cc = vp[vp.variant.isin(["t1_v3_strict_nocooc", "t1_v3_temporal_nocooc"])]
num("NocoocMaxAbs", f"{cc.delta.abs().max():.3f}")
num("NocoocNModels", str(cc.model.nunique()))
write("app_variants.tex", tab("lcccc", "Training-graph variant & $\\Delta$ degree prior & $\\Delta$ gap to prior (models) & "
                                       "gap CI $<0$ / $>0$ & better/worse/learned", lines))

# candidate pools
pool_lines = []
for pool, tag in [("all", ""), ("target_all", "_target_all"), ("target_train", "_target_train")]:
    mt = pd.read_csv(A3 / f"main_table{tag}.csv").set_index(["setting", "model"])
    vv = pd.read_csv(A3 / f"vs_degree{tag}.csv").set_index(["setting", "model"])
    cells = []
    for s in TRANSFER:
        if (s, "popularity") not in mt.index:
            cells += ["--", "--"]
            continue
        q = vv.loc[[i for i in vv.index if i[0] == s and i[1] in LEARNED]]
        cells += [f"{mt.loc[(s, 'popularity'), 'mrr']:.3f}",
                  f"{int((q.mrr_label == 'better').sum())}/{int((q.mrr_label == 'inconclusive').sum()) + int((q.mrr_label == 'equivalent').sum())}/{int((q.mrr_label == 'worse').sum())}"]
    nm = {"all": "All seen entities (default)", "target_all": "Entities with a target edge in the graph",
          "target_train": "Entities with a target edge in training"}[pool]
    pool_lines.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    if pool == "target_train":
        num("PoolTrainPriorStrict", f"{mt.loc[('t1_study_strict', 'popularity'), 'mrr']:.3f}")
        num("PoolTrainPriorExt", f"{mt.loc[('t1_external', 'popularity'), 'mrr']:.3f}")
        q = vv.loc[[i for i in vv.index if i[0] in TRANSFER and i[1] in LEARNED]]
        num("PoolTrainBetter", str(int((q.mrr_label == "better").sum())))
hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{HEAD[s]}}}" for s in TRANSFER)
cmr = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(TRANSFER)))
write("app_pools.tex", tab("l" + "rr" * len(TRANSFER), "Candidate pool & " + " & ".join(["prior", "b/i/w"] * len(TRANSFER)),
                           pool_lines, pre=" & " + hdr + " \\\\\n" + cmr + "\n"))

# ------------------------------------------------------------------------------------------ P1-1 same-study control
pc = pd.read_csv(A3 / "p11_contrasts.csv").set_index(["model", "contrast"])
pp = pd.read_csv(A3 / "p11_permutation.csv").set_index("model")
lines = []
for m in [x for x in ROWS if (x, "same_minus_other") in pc.index]:
    cells = []
    for c in ["same_minus_base", "other_minus_base", "same_minus_other"]:
        r = pc.loc[(m, c)]
        t = f"{r.delta:+.4f} {{\\tiny{ci(r.lo95, r.hi95, 4)}}}"
        cells.append(f"\\textbf{{{t}}}" if (r.lo95 > 0 or r.hi95 < 0) else t)
    r = pp.loc[m]
    cells.append(f"{r.rho:.2f} ({r.p_perm:.4f})")
    lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    k = NAME[m].split()[0].replace("-", "")
    r = pc.loc[(m, "same_minus_other")]
    num(f"PSame{k}", f"{r.delta:+.4f}")
    num(f"PSame{k}CI", ci(r.lo95, r.hi95, 4))
    num(f"PRho{k}", f"{pp.loc[m].rho:.2f}")
    num(f"PPerm{k}", f"{pp.loc[m].p_perm:.4f}")
write("app_p11.tex", tab("lcccc", "Method & Same $-$ base & Other $-$ base & Same $-$ other & $\\rho$ (perm.\\ $p$)", lines))
num("PStudies", f"{int(pp.n_studies.iloc[0]):,}")

# ------------------------------------------------------------------------------------------ P1-2 learnability
rep = pd.read_csv(A3 / "replication_rate.csv").iloc[0]
num("ReplAny", f"{100 * rep.replicated_any:.1f}")
num("ReplDir", f"{100 * rep.replicated_same_direction:.1f}")
num("ReplPairs", f"{int(rep.pairs):,}")
num("ReplDiseases", f"{int(rep.diseases):,}")
lines = []
SUB = [("t1_study_strict", "replicated_2plus_studies", "Strict, reported by $\\geq$2 studies"),
       ("t1_study_strict", "rereported_after_snapshot", "Strict, re-reported after the cut-off"),
       ("t1_temporal", "replicated_2plus_studies", "Temporal, reported by $\\geq$2 studies"),
       ("t1_temporal", "rereported_after_snapshot", "Temporal, re-reported after the cut-off"),
       ("t1_external", "published_after_snapshot", "Out-of-time, published after the cut-off"),
       ("t1_external", "published_2025_or_later", "Out-of-time, published 2025 or later"),
       ("t1_external", "reported_by_2plus_new_studies", "Out-of-time, reported by $\\geq$2 new studies")]
for s, sub, nm in SUB:
    f = A3 / (f"subsets_learnability_{s}.csv" if s != "t1_external" else "subsets_external.csv")
    d = pd.read_csv(f)
    d = d[d.subset == sub]
    if not len(d):
        continue
    pr = d[d.model == "popularity"].iloc[0]
    q = d[d.model.isin(LEARNED)]
    better = ", ".join(f"{NAME[m]} {r.delta:+.3f}" for m, r in q.set_index("model").iterrows() if r.label == "better")
    lines.append(f"{nm} & {pr.n_test:,.0f} & {pr.mrr:.3f} & {int((q.label == 'better').sum())} & "
                 f"{int(q.label.isin(['inconclusive', 'equivalent']).sum())} & {int((q.label == 'worse').sum())} & "
                 f"{q.mde80.min():.3f}--{q.mde80.max():.3f} & {better or '--'} \\\\")
    key = {"replicated_2plus_studies": "Repl", "rereported_after_snapshot": "Rerep", "published_after_snapshot": "After",
           "published_2025_or_later": "Late", "reported_by_2plus_new_studies": "TwoNew"}[sub] + \
        {"t1_study_strict": "Strict", "t1_temporal": "Temporal", "t1_external": "Ext"}[s]
    num(f"N{key}", f"{pr.n_test:,.0f}")
    num(f"Better{key}", str(int((q.label == "better").sum())))
    for m, r in q.set_index("model").iterrows():
        if r.label == "better":
            num(f"{key}{NAME[m].split()[0].replace('-', '')}", f"{r.delta:+.3f} {ci(r.lo95, r.hi95)}")
l3 = pd.read_csv(A3 / "subsets_learnability_t1_temporal.csv").set_index(["subset", "model"])
if ("replicated_2plus_studies", "l3") in l3.index:
    r = l3.loc[("replicated_2plus_studies", "l3")]
    num("ReplTemporalLthree", f"{r.delta:+.3f} {ci(r.lo95, r.hi95)}")
write("app_learnability.tex", tab("lrrrrrrp{3.2cm}", "Subset & Triples & Prior MRR & Better & Incon. & Worse & MDE$_{80}$ & "
                                                     "Better than the prior", lines))

# ------------------------------------------------------------------------------------------ cross-source stress test
pr = pd.read_csv(A3 / "probe_v3.csv")
PV = [("P_vs_allunknown", "P/U"), ("P_vs_allunknown_degree_matched", "P/U$_{\\deg}$"), ("P_vs_N_within", "P/N"),
      ("P_vs_N_tested", "P$_{t}$/N"), ("P_vs_N_tested_gut", "P$_{t,\\text{gut}}$/N"),
      ("P_vs_N_det_degree_matched", "P$_{t}$/N$_{\\text{m}}$"),
      ("N_vs_allunknown", "N/U")]
P = pr.set_index(["setting", "model", "variant"])
for s, fname in [("t1_study_strict", "app_probe_strict.tex"), ("t1_study", "app_probe_study.tex"),
                 ("t1_temporal", "app_probe_temporal.tex"),
                 ("t1_external", "app_probe_external.tex")]:
    lines = []
    for m in [x for x in ROWS if (s, x, "P_vs_N_within") in P.index]:
        cells = []
        for v, _ in PV:
            if (s, m, v) not in P.index:
                cells.append("--")
                continue
            r = P.loc[(s, m, v)]
            t = f"{r.auroc:.2f}"
            cells.append(f"\\textbf{{{t}}}" if r.hi < 0.5 else (f"\\textit{{{t}}}" if r.lo > 0.5 else t))
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
    write(fname, tab("l" + "r" * len(PV), "Method & " + " & ".join(h for _, h in PV), lines))
pe = [m for m in ["cn", "ra", "l3", "transe", "distmult", "complex", "rotate", "rgcn"]]
for s, k in [("t1_study_strict", "Strict"), ("t1_temporal", "Temporal")]:
    for v, kk in [("P_vs_N_within", "Within"), ("P_vs_N_tested", "Tested"), ("P_vs_N_det_degree_matched", "Matched"),
                  ("N_vs_allunknown", "NU"), ("P_vs_allunknown", "PU")]:
        q = pr[(pr.setting == s) & (pr.variant == v) & pr.model.isin(pe)]
        if len(q):
            num(f"Pr{k}{kk}Lo", f"{q.auroc.min():.2f}")
            num(f"Pr{k}{kk}Hi", f"{q.auroc.max():.2f}")
            num(f"Pr{k}{kk}Below", str(int((q.hi < 0.5).sum())))
            num(f"Pr{k}{kk}Above", str(int((q.lo > 0.5).sum())))
            num(f"Pr{k}{kk}N", str(len(q)))
    if (s, "gbm", "P_vs_N_det_degree_matched") in P.index:
        r = P.loc[(s, "gbm", "P_vs_N_det_degree_matched")]
        num(f"Pr{k}Gbm", f"{r.auroc:.2f} {ci(r.lo, r.hi, 2)}")
# all regimes x path/embedding models: P vs N within, below 0.5
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_within") & pr.model.isin(pe)]
num("PrAllWithinLo", f"{q.auroc.min():.2f}")
num("PrAllWithinHi", f"{q.auroc.max():.2f}")
num("PrAllWithinBelow", str(int((q.hi < 0.5).sum())))
num("PrAllWithinN", str(len(q)))
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_det_degree_matched") & pr.model.isin(pe)]
num("PrAllMatchedLo", f"{q.auroc.min():.2f}")
num("PrAllMatchedHi", f"{q.auroc.max():.2f}")
num("PrAllMatchedBelow", str(int((q.hi < 0.5).sum())))
num("PrAllMatchedAbove", str(int((q.lo > 0.5).sum())))
num("PrAllMatchedN", str(len(q)))
pe2 = ["l3", "transe", "distmult", "complex", "rotate", "rgcn", "ncn", "ultra-zeroshot"]
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "N_vs_allunknown") & pr.model.isin(pe2)]
num("PrNULo", f"{q.auroc.min():.2f}")
num("PrNUHi", f"{q.auroc.max():.2f}")
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_allunknown") & pr.model.isin(pe2)]
num("PrPULo", f"{q.auroc.min():.2f}")
num("PrPUHi", f"{q.auroc.max():.2f}")
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_tested") & pr.model.isin(pe)]
num("PrAllTestedLo", f"{q.auroc.min():.2f}")
num("PrAllTestedHi", f"{q.auroc.max():.2f}")
grid = pr[pr.variant.str.startswith("grid_") & pr.setting.isin(["t1_study_strict"]) & pr.model.isin(pe)]
num("GridLo", f"{grid.auroc.min():.2f}")
num("GridHi", f"{grid.auroc.max():.2f}")
num("GridBelow", str(int((grid.hi < 0.5).sum())))
num("GridN", str(len(grid)))
gl = []
for m in [x for x in ROWS if x in set(grid.model)]:
    g_ = pr[pr.variant.str.startswith("grid_") & (pr.setting == "t1_study_strict") & (pr.model == m)]
    gl.append(f"{NAME[m]} & {g_.auroc.min():.2f}--{g_.auroc.max():.2f} & {g_.hi.max():.2f} & {int((g_.hi < 0.5).sum())}/{len(g_)} \\\\")
write("app_probe_grid.tex", tab("lccc", "Method & AUROC(P,N) range & max upper 95\\% & CI $<0.5$", gl))
dg = pd.read_csv(A3 / "probe_v3_diagnostics.csv")
dg = dg[dg.setting == "t1_study_strict"].groupby("model").mean(numeric_only=True)
num("ShareTested", f"{100 * dg.share_P_tested.mean():.1f}")
num("DegP", f"{dg.mean_log_deg_P.mean():.2f}")
num("DegN", f"{dg.mean_log_deg_N.mean():.2f}")
for m in ["tucker", "rotate", "l3"]:
    if m in dg.index:
        num(f"Rho{NAME[m].split()[0].replace('-', '')}", f"{dg.loc[m, 'score_degree_spearman']:.2f}")
oth = dg[dg.index.isin(["l3", "transe", "distmult", "complex", "rotate", "rgcn", "ultra-zeroshot"])].score_degree_spearman
num("RhoOtherLo", f"{oth.min():.2f}")
num("RhoOtherHi", f"{oth.max():.2f}")
fw = pr[(pr.variant == "P_vs_N_within_fwd") & pr.setting.isin(TRANSFER) & (pr.model == "ultra-zeroshot")]
if len(fw):
    num("UltraFwdLo", f"{fw.auroc.min():.2f}")
    num("UltraFwdHi", f"{fw.auroc.max():.2f}")
nn = pr[(pr.variant == "P_vs_N_within") & (pr.model == "popularity")].set_index("setting")
num("ProbeDis", f"{nn.loc['t1_study_strict', 'n_diseases']:.0f}")

# ------------------------------------------------------------------------------------------ v2 analyses kept as is
dec = pd.read_csv(A2 / "decomposition.csv").set_index(["model", "quantity"])
Q = [("pair_all", "Pair"), ("pair_matched_rw", "Pair$_{\\text{SL}}$"), ("study_reveal", "Study+R"),
     ("study", "Study-excl."), ("study_strict", "Strict"), ("composition_effect", "$\\Delta_{\\text{comp}}$"),
     ("holdout_effect", "$\\Delta_{\\text{res}}$"), ("sibling_effect", "$\\Delta_{\\text{sib}}$"),
     ("cohort_backdoor", "$\\Delta_{\\text{co}}$"), ("share_composition", "comp.\\ share")]
lines = []
for m in [x for x in ROWS if (x, "pair_all") in dec.index]:
    cells = []
    for q, _ in Q:
        if (m, q) not in dec.index:
            cells.append("--")
            continue
        r = dec.loc[(m, q)]
        if q == "share_composition":
            cells.append(f"{100 * r.estimate:.0f}\\%")
        elif q.endswith("effect") or q == "cohort_backdoor":
            t = f"{r.estimate:.3f}"
            cells.append(t if (r.lo > 0 or r.hi < 0) else f"\\textit{{{t}}}")
        else:
            cells.append(f"{r.estimate:.3f}")
    lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
write("app_decomposition.tex", tab("l" + "r" * len(Q), "Method & " + " & ".join(h for _, h in Q), lines))
sh = dec.xs("share_composition", level="quantity").estimate
sh = sh[sh.index != "random"]
num("CompLo", f"{100 * sh.min():.0f}")
num("CompHi", f"{100 * sh.max():.0f}")
he = dec.xs("holdout_effect", level="quantity")
num("ResPrior", f"{he.loc['popularity', 'estimate']:.3f} {ci(he.loc['popularity', 'lo'], he.loc['popularity', 'hi'])}")
hl = he[he.index.isin(EMB + ["rgcn", "ultra-zeroshot"])]
num("ResLearnedLo", f"{hl.estimate.min():.3f}")
num("ResLearnedHi", f"{hl.estimate.max():.3f}")
num("ResLearnedSig", str(int((hl.lo > 0).sum())))
num("ResLearnedN", str(len(hl)))
se = dec.xs("sibling_effect", level="quantity")
for m in ["rgcn", "ultra-zeroshot", "rotate", "l3", "popularity"]:
    if m in se.index:
        num(f"Sib{NAME[m].split()[0].replace('-', '').replace('(', '')}", f"{se.loc[m, 'estimate']:.3f}")

cl = pd.read_csv(A2 / "closure.csv").set_index(["setting", "model", "quantity"])
CR = ["t1_pair", "t1_study", "t1_study_strict", "t1_temporal"]
lines = []
for m in [x for x in ROWS if x != "popularity" and any((s, x, "vsdeg_closure") in cl.index for s in CR)]:
    cells = []
    for s in CR:
        for q in ["vsdeg_closure", "vsdeg_noclosure_std"]:
            if (s, m, q) not in cl.index:
                cells.append("--")
                continue
            r = cl.loc[(s, m, q)]
            t = f"{r.estimate:+.3f}"
            cells.append(f"\\textbf{{{t}}}" if (r.lo > 0 or r.hi < 0) else t)
    lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
share = " & ".join(f"\\multicolumn{{2}}{{c}}{{{100 * cl.loc[(s, 'popularity', 'share_noclosure'), 'estimate']:.0f}\\%}}"
                   for s in CR)
hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{{HEAD[s]}}}" for s in CR)
cmr = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(CR)))
write("app_closure.tex", tab("l" + "rr" * len(CR), " & " + " & ".join(["closure", "none (std.)"] * len(CR)),
                             ["Share without closure & " + share + " \\\\", "\\midrule"] + lines,
                             pre=" & " + hdr + " \\\\\n" + cmr + "\n"))

cs = pd.read_csv(A2 / "cold_sides.csv").set_index(["setting", "model", "side"])
lines = []
for m in [x for x in ROWS + ["random"] if any((s, x, "cold_entity_is_query") in cs.index for s in ["t1_cold_microbe"])]:
    cells = [f"{cs.loc[(s, m, sd), 'mrr']:.3f}" if (s, m, sd) in cs.index else "--"
             for s in ["t1_cold_microbe", "t1_cold_disease"] for sd in ["cold_entity_is_query", "cold_entity_is_target"]]
    lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
write("app_cold.tex", tab("lrrrr", "Method & new query & new answer & new query & new answer", lines,
                          pre=" & \\multicolumn{2}{c}{Cold microbe} & \\multicolumn{2}{c}{Cold disease} \\\\\n"
                              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"))
qq = cs.xs("cold_entity_is_target", level="side").loc["t1_cold_microbe"].mrr
num("ColdMicAnswerMax", f"{qq.max():.3f}")
num("ColdMicAnswerLearnedMax", f"{qq[qq.index.isin(LEARNED)].max():.3f}")
num("ColdMicQueryPrior", f"{cs.loc[('t1_cold_microbe', 'popularity', 'cold_entity_is_query'), 'mrr']:.3f}")
num("ColdMicQueryTucker", f"{cs.loc[('t1_cold_microbe', 'tucker', 'cold_entity_is_query'), 'mrr']:.3f}")

# classic protocol audit (+ KGNMDA)
au = pd.read_csv(R / ".clawsgo/figures/fig4-classic-audit/code/data/audit_summary.csv")
au = au[au.kernel_mode == "train"].set_index(["dataset", "method", "protocol"])
kg = pd.read_csv(R / "audit_hmdad/kgnmda/results.csv").set_index("protocol")
LIT = {"KATZHMDA": "0.838", "LRLSHMDA": "0.891", "BiRWHMDA": "0.896", "Degree": "--"}
lines = []
for ds in ["HMDAD", "Disbiome"]:
    for m in ["Degree", "KATZHMDA", "LRLSHMDA", "BiRWHMDA"]:
        c = [ds if m == "Degree" else "", "Degree prior" if m == "Degree" else m, LIT[m] if ds == "HMDAD" else "--"]
        for pr_, col in [("LOO", "auroc"), ("A", "auroc"), ("B", "auroc"), ("C", "auroc"), ("A", "auroc_matched")]:
            r = au.loc[(ds, m, pr_)]
            v, sd_ = r[f"{col}_mean"], r[f"{col}_sd"]
            c.append(f"{v:.3f}" if pr_ == "LOO" else f"{v:.3f}\\,{{\\scriptsize$\\pm${sd_:.3f}}}")
        lines.append(" & ".join(c) + " \\\\")
    if ds == "HMDAD":
        c = ["", "KGNMDA", "--", "--"] + [f"{kg.loc[p_, 'auroc_mean']:.3f}\\,{{\\scriptsize$\\pm${kg.loc[p_, 'auroc_sd']:.3f}}}"
                                          for p_ in ["A", "B", "C"]] + [f"{kg.loc['A', 'auroc_matched_mean']:.3f}"]
        lines.append(" & ".join(c) + " \\\\")
        lines.append("\\midrule")
write("app_hmdad.tex", tab("llcccccc", "Data & Method & Published & LOOCV & Random 5-fold & Study-grouped & Cold microbe & "
                                       "Matched neg.", lines))
for p_, k in [("A", "A"), ("B", "B"), ("C", "C")]:
    num(f"Kg{k}", f"{kg.loc[p_, 'auroc_mean']:.3f}")
num("KgMatchedLo", f"{kg.auroc_matched_mean.min():.2f}")
num("KgMatchedHi", f"{kg.auroc_matched_mean.max():.2f}")
h = au.xs("HMDAD", level="dataset")
cls = ["KATZHMDA", "LRLSHMDA", "BiRWHMDA"]
num("HmA", f"{min(h.loc[(m, 'A'), 'auroc_mean'] for m in cls):.2f}--{max(h.loc[(m, 'A'), 'auroc_mean'] for m in cls):.2f}")
num("HmB", f"{min(h.loc[(m, 'B'), 'auroc_mean'] for m in cls):.2f}--{max(h.loc[(m, 'B'), 'auroc_mean'] for m in cls):.2f}")
num("HmC", f"{min(h.loc[(m, 'C'), 'auroc_mean'] for m in cls):.2f}--{max(h.loc[(m, 'C'), 'auroc_mean'] for m in cls):.2f}")
mv = [h.loc[(m, 'A'), 'auroc_matched_mean'] for m in cls] + [kg.loc["A", "auroc_matched_mean"]]
num("HmM", f"{min(mv):.3f}--{max(mv):.3f}")
d_ = au.xs("Disbiome", level="dataset")
num("DbPriorA", f"{d_.loc[('Degree', 'A'), 'auroc_mean']:.3f}")
num("DbLrlsA", f"{d_.loc[('LRLSHMDA', 'A'), 'auroc_mean']:.3f}")
num("DbLrlsB", f"{d_.loc[('LRLSHMDA', 'B'), 'auroc_mean']:.3f}")
num("DbPriorB", f"{d_.loc[('Degree', 'B'), 'auroc_mean']:.3f}")

# record layer (appendix)
st = pd.read_csv(REC / "disagreement_strata.csv")
lg = pd.read_csv(REC / "disagreement_logit.csv").rename(columns={"Unnamed: 0": "term"}).set_index("term")
dr = pd.read_csv(REC / "direction_summary.csv").set_index("model")
lines = []
for v, nm in [("same_site", "Body site (system level)"), ("same_study", "Study"), ("same_platform", "Platform"),
              ("same_source", "Source database")]:
    a = st[(st.variable == v) & (st.value == 0)].iloc[0]
    b = st[(st.variable == v) & (st.value == 1)].iloc[0]
    o = f"{lg.loc[v, 'or']:.2f} [{lg.loc[v, 'or_lo']:.2f}, {lg.loc[v, 'or_hi']:.2f}]" if v in lg.index else "--"
    lines.append(f"{nm} & {100 * a.rate:.1f}\\% ({int(a.record_pairs):,}) & {100 * b.rate:.1f}\\% ({int(b.record_pairs):,}) & {o} \\\\")
write("app_conflict.tex", tab("lccc", "Attribute & Differs: conflict rate ($n$) & Shared: conflict rate ($n$) & "
                                      "Adjusted OR [95\\% CI]", lines))
DN = [("majority", "Majority class"), ("prior_disease", "Disease prior"), ("prior_family", "Family prior"),
      ("prior_genus", "Genus prior"), ("prior_microbe", "Microbe prior"), ("gbm_context", "GBM, context only"),
      ("gbm_context_priors", "GBM, context + priors")]
write("app_direction.tex", tab("lccc", "Model & AUROC & SD over seeds & Study-cluster 95\\% CI",
                               [f"{nm} & {dr.loc[k, 'auroc']:.3f} & {dr.loc[k, 'sd']:.3f} & "
                                f"[{dr.loc[k, 'lo']:.3f}, {dr.loc[k, 'hi']:.3f}] \\\\" for k, nm in DN]))
ra = json.load(open(R / "results2/records_v2_audit.json"))


LABEL_MAP = {"Lit44": "Literature extraction tables"}


def srclab(s):
    """Source label as printed in the paper: upstream release suffixes (_v2, _v3) are dropped."""
    t = __import__("re").sub(r"_v\d+(\.\d+)?$", "", str(s))
    return LABEL_MAP.get(t, t).replace("_", "\\_")


lines = [f"{srclab(c['record_source'])} & {c['records']:,} & " + " & ".join(
    f"{100 * c[k]:.0f}" for k in ["pmid", "year", "site", "platform", "n"]) + " \\\\"
         for c in sorted(ra["coverage"], key=lambda r: -r["records"])]
write("app_records.tex", tab("lrrrrrr", "Source & Records & PMID & Year & Site & Platform & $n$", lines))

# context variants
lines = []
for m in ["popularity", "cn", "ra", "l3", "rotate", "tucker"]:
    cells = []
    for s in ["t1_study", "t1_study_ctx_open"]:
        if (s, m) not in main.index:
            cells.append("--")
            continue
        r = main.loc[(s, m)]
        cells.append(f"{r.mrr:.3f} {{\\tiny{ci(r.mrr_lo, r.mrr_hi)}}} ({int(r.n_seeds)})")
    lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
write("app_context.tex", tab("lcc", "Method & Study-excl. & Study-excl.\\ + open context", lines))
if ("t1_study_ctx_open", "popularity") in main.index:
    num("CtxPrior", f"{main.loc[('t1_study_ctx_open', 'popularity'), 'mrr']:.3f}")

# split sizes
aud = json.load(open(R / "results2/build_audit.json"))["seeds"]


def mm(k, kk):
    v = [aud[s][k][kk] for s in aud]
    return f"{np.mean(v):,.0f}" if min(v) == max(v) else f"{np.mean(v):,.0f} [{min(v):,}--{max(v):,}]"


ROWS_S = [("Pair", "pair", "train", "valid", "test", "--"), ("Study-excl.", "study", "train", "valid", "test", "revealed"),
          ("Strict", "study", "strict_train", "valid", "test", "dropped_by_strict"),
          ("Temporal", "t1_temporal", "train", "valid", "test", "--"),
          ("Temporal\\,$-$Dis.", "t1_temporal_nodis", "train", "valid", "test", "removed"),
          ("Cold microbe", "t1_cold_microbe", "train", "valid", "test", "cold_test"),
          ("Cold disease", "t1_cold_disease", "train", "valid", "test", "cold_test")]
TARGET_RELS = ["enriched_in", "depleted_in"]


def cold_pairs(setting):
    """Distinct (microbe, disease) target pairs per split part and seed, from the released split tables.
    The build audit counts triples (a pair with both directions counts twice), so the cold rows are recounted here."""
    out = {}
    for sd in [42, 123, 456, 789, 2024]:
        row = {}
        for part in ["train", "valid", "test"]:
            df = pd.read_parquet(R / f"release/v3/benchmark/splits/{setting}/seed_{sd}_{part}.parquet",
                                 columns=["head_id", "relation", "tail_id"])
            df = df[df.relation.isin(TARGET_RELS)]
            row[part] = len(df[["head_id", "tail_id"]].drop_duplicates())
        out[sd] = row
    return out


def mm_pairs(cp, part):
    v = [cp[s][part] for s in cp]
    return f"{np.mean(v):,.0f}" if min(v) == max(v) else f"{np.mean(v):,.0f} [{min(v):,}--{max(v):,}]"


CP = {k: cold_pairs(k) for k in ["t1_cold_microbe", "t1_cold_disease"]}
N_ALL_PAIRS = {sum(aud[s]["pair"][p] for p in ("train", "valid", "test")) for s in aud}
assert len(N_ALL_PAIRS) == 1, N_ALL_PAIRS
for k, cp in CP.items():     # every regime partitions the same set of target pairs
    assert all(sum(r.values()) == next(iter(N_ALL_PAIRS)) for r in cp.values()), (k, cp, N_ALL_PAIRS)


def split_row(n_, k, tr, va, te, ex):
    if k in CP:
        cells = [mm_pairs(CP[k], p) for p in ("train", "valid", "test")]
    else:
        cells = [mm(k, tr), mm(k, va), mm(k, te)]
    return f"{n_} & {cells[0]} & {cells[1]} & {cells[2]} & {'--' if ex == '--' else mm(k, ex)} \\\\"


write("app_splits.tex", tab("lrrrr", "Regime & Train pairs & Valid pairs & Test pairs & Extra",
                            [split_row(*r) for r in ROWS_S]))
num("StrictDropped", f"{np.mean([aud[s]['study']['dropped_by_strict'] for s in aud]):,.0f}")

# source terms
terms = json.load(open(R / "release/v3/SOURCE_TERMS.json"))
lines = []
for t in terms["sources"]:
    lab, n_, st_ = t.get("source"), t.get("current_labelled_assertions") or 0, t.get("status", "")
    ob = t.get("license_observation") or ""
    st_ = st_.replace("pending_snapshot", "pending").replace("restricted_pending", "restricted, pending")
    ob = ob.replace("snapshot license confirmation pending", "release licence confirmation pending")
    lines.append(f"{srclab(lab)} & {int(n_):,} & {st_.replace('_', ' ')} & {{\\scriptsize {ob.replace('&', '\\&')}}} \\\\")
write("app_terms.tex", tab("lrlp{6.3cm}", "Source label & Assertions & Status & Terms observation", lines))

# record-level semantic audit of 500 sampled records (counts supplied by the authors: results3/analysis/semantic_audit_500.csv)
au5 = pd.read_csv(A3 / "semantic_audit_500.csv")
assert (au5.audited == au5.supported + au5.incorrect + au5.unverifiable).all(), au5
au5["group"] = au5.source.replace({"Curated-table literature extractions": "Other literature sources"})
ag = au5.groupby("group", sort=False)[["audited", "supported", "incorrect", "unverifiable"]].sum()
lines = []
for g_, r in ag.iterrows():
    lines.append(f"{g_} & {int(r.audited)} & {int(r.supported)} & {int(r.incorrect)} & {int(r.unverifiable)} & "
                 f"{100 * r.supported / r.audited:.1f} \\\\")
    key = {"BugSigDB": "Bsdb", "Disbiome": "Disb", "Other literature sources": "Oth"}[g_]
    num(f"Audit{key}N", f"{int(r.audited)}")
    num(f"Audit{key}Pct", f"{100 * r.supported / r.audited:.1f}")
T = ag.sum()
lines.append("\\midrule")
lines.append(f"Total & {int(T.audited)} & {int(T.supported)} & {int(T.incorrect)} & {int(T.unverifiable)} & "
             f"{100 * T.supported / T.audited:.1f} \\\\")
write("app_semantic_audit.tex", tab("lrrrrr", "Source & Audited & Supported & Incorrect & Unverifiable & Supported (\\%)", lines))
num("AuditN", f"{int(T.audited)}")
num("AuditK", f"{int(T.supported)}")
num("AuditPct", f"{100 * T.supported / T.audited:.1f}")
num("AuditBad", f"{int(T.incorrect)}")
num("AuditBadPct", f"{100 * T.incorrect / T.audited:.1f}")
num("AuditUnv", f"{int(T.unverifiable)}")
num("AuditUnvPct", f"{100 * T.unverifiable / T.audited:.1f}")

# independence audit (PMID -> BioProject)
ob = pd.read_csv(R / "results3/indep/overlap_by_seed.csv")
t1 = json.load(open(R / "results3/indep/t1_stats.json"))
a_ = ob[ob.tier == "all_sources"]
e_ = ob[ob.tier == "elink_only"]
num("BpMapped", f"{t1['pmids_with_bioproject']:,}")
num("BpPmids", f"{t1['pmid_studies_mappable']:,}")
num("BpRate", f"{100 * t1['mapping_rate_of_mappable_pmids']:.1f}")
num("BpOverlapLo", str(int(a_.distinct_bioprojects_overlap.min())))
num("BpOverlapHi", str(int(a_.distinct_bioprojects_overlap.max())))
num("BpElinkSeeds", str(int((e_.affected_case_contrasts > 0).sum())))
num("BpHealthyLo", f"{100 * a_.heldout_healthy_run_frac.min():.2f}")
num("BpHealthyHi", f"{100 * a_.heldout_healthy_run_frac.max():.2f}")

# reference regimes and named exceptions outside the four transfer regimes
q = vs.loc[[i for i in vs.index if i[0] == "t1_pair" and i[1] in LEARNED]]
num("PairBetter", str(int((q.mrr_label == "better").sum())))
num("PairN", str(len(q)))
if ("t1_cold_disease", "ultra-zeroshot") in vs.index:
    r = vs.loc[("t1_cold_disease", "ultra-zeroshot")]
    num("ColdDisUltra", f"{r.mrr_delta:+.4f} {ci(r.mrr_lo95, r.mrr_hi95, 4)}")
    num("ColdDisUltraLabel", r.mrr_label)
q = vs.loc[[i for i in vs.index if i[0] == "t1_cold_disease" and i[1] in LEARNED]]
num("ColdDisBetter", str(int((q.mrr_label == "better").sum())))
# P1-1 matched pairs per seed and BioProject-overlap removals (released build audit)
p11 = [pd.read_parquet(R / f"release/v3/benchmark/p11_pairs/seed_{sd}.parquet") for sd in [42, 123, 456, 789, 2024]]
mt_ = [int((d.set == "matched").sum()) for d in p11]
rv_ = [int((d.set == "revealed").sum()) for d in p11]
num("PMatchedLo", f"{min(mt_):,}")
num("PMatchedHi", f"{max(mt_):,}")
num("PRevealedLo", f"{min(rv_):,}")
num("PRevealedHi", f"{max(rv_):,}")
ba = json.load(open(R / "release/v3/benchmark/variants_build_audit.json"))
rm_ = [v["removed_target_pairs"] for v in ba["t1_v3_strict_nooverlap"].values()]
num("BpRemovedLo", f"{min(rm_):,}")
num("BpRemovedHi", f"{max(rm_):,}")
rm_ = [v["removed_target_pairs"] for v in ba["t1_v3_strict_nocohort"].values()]
num("NocohortRemoved", f"{rm_[0]:,}")
num("NocohortCooc", f"{list(ba['t1_v3_strict_nocohort'].values())[0]['removed_cooc']:,}")
rm_ = [v["removed_target_pairs"] for v in ba["t1_v3_temporal_dated"].values()]
num("DatedRemoved", f"{rm_[0]:,}")
# random-search tuning summary (filled once the final cells exist)
for m in RS:
    for s_ in ["t1_study_strict", "t1_external"]:
        if (s_, m) in vs.index:
            r = vs.loc[(s_, m)]
            num(f"Rs{m.split('-')[0].capitalize()}{'Strict' if s_ == 't1_study_strict' else 'Ext'}",
                f"{r.mrr_delta:+.3f} {ci(r.mrr_lo95, r.mrr_hi95)} ({r.mrr_label})")

# ------------------------------------------------------------------------------------------ Table 3: robustness
def counts(labels):
    labels = list(labels)
    return (sum(l == "better" for l in labels), sum(l in ("inconclusive", "equivalent") for l in labels),
            sum(l == "worse" for l in labels))


def rrow(name, prior, labels, base_prior=None):
    b, i, w = counts(labels)
    d = "" if base_prior is None or pd.isna(prior) else f" ({prior - base_prior:+.3f})"
    pt = "--" if pd.isna(prior) else f"{prior:.3f}{d}"
    return f"{name} & {pt} & {b} & {i} & {w} \\\\" if len(labels) else f"{name} & {pt} & \\multicolumn{{3}}{{c}}{{pending}} \\\\"


mt_tr = pd.read_csv(A3 / "main_table_target_train.csv").set_index(["setting", "model"])
vs_tr = pd.read_csv(A3 / "vs_degree_target_train.csv").set_index(["setting", "model"])
learned_labels = lambda V, s_, mods=LEARNED: [V.loc[(s_, m), "mrr_label"] for m in mods if (s_, m) in V.index]
prior = lambda M, s_: M.loc[(s_, "popularity"), "mrr"] if (s_, "popularity") in M.index else np.nan
rob = []
for base, block in [("t1_study_strict", "Strict"), ("t1_temporal", "Temporal"), ("t1_external", "Out-of-time")]:
    bp = prior(main, base)
    rob.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{block}}}}} \\\\")
    rob.append(rrow("\\quad Default protocol", bp, learned_labels(vs, base)))
    rob.append(rrow("\\quad Candidates: entities with a training target edge", prior(mt_tr, base),
                    learned_labels(vs_tr, base), bp))
    if base == "t1_study_strict":
        for v in ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc"]:
            rob.append(rrow("\\quad Training graph: " + HEAD[v].replace("Strict $-$ ", "$-$ "), prior(main, v),
                            learned_labels(vs, v), bp))
    if base == "t1_temporal":
        for v in ["t1_v3_temporal_dated", "t1_v3_temporal_nocooc"]:
            nm = "cohort support dated $\\leq$2021" if v.endswith("dated") else "$-$ co-occurrence"
            rob.append(rrow("\\quad Training graph: " + nm, prior(main, v), learned_labels(vs, v), bp))
    if base == "t1_external":
        e = pd.read_csv(A3 / "subsets_external.csv")
        e = e[e.subset == "published_after_snapshot"].set_index("model")
        rob.append(rrow("\\quad Test: studies published after the cut-off", e.loc["popularity", "mrr"],
                        [e.loc[m, "label"] for m in LEARNED if m in e.index], bp))
    if base in ("t1_study_strict", "t1_external"):
        rob.append(rrow("\\quad Tuning: random search (" + ("RotatE, TuckER, R-GCN" if base == "t1_study_strict"
                        else "RotatE, TuckER") + ")", bp, learned_labels(vs, base, RS)))
write("robustness.tex", tab("lrrrr", "Check & Prior MRR & Better & Incon./equiv. & Worse", rob))

# dataset size macros (released pair support)
ps_ = pd.read_parquet(R / "release/v3/benchmark/pair_support.parquet")
coh = ps_.cohort_studies.str.split(";").explode()
coh = coh[coh != ""]
lit = ps_.lit_studies.str.split(";").explode()
lit = lit[lit != ""]
num("NPairs", f"{len(ps_):,}")
num("NLitStudies", f"{lit.nunique():,}")
num("NGMContrasts", f"{coh[coh.str.startswith('GMREPO')].nunique():,}")
num("NMBGroups", f"{coh[coh.str.startswith('MBODYMAP')].nunique():,}")
num("NCohortOnly", f"{int(((ps_.cohort_studies != '') & (ps_.lit_studies == '')).sum()):,}")
num("NCohortAny", f"{int((ps_.cohort_studies != '').sum()):,}")

# out-of-time publication dates (PubMed; two DOI-keyed studies dated by Crossref, both after the cut-off)
dt = pd.read_csv(R / "results3/indep/external_pmid_dates.tsv", sep="\t", dtype={"pmid": str})
dt = dt[dt.pmid != "10"]
num("ExtPmids", f"{dt.pmid.nunique():,}")
num("ExtLatePmids", str(int((pd.to_numeric(dt.earliest_year, errors="coerce") >= 2025).sum())))
num("ExtAfterPmids", str(int((dt.after_cutoff.astype(str) == "True").sum())))
q = pr[(pr.setting == "t1_study_strict") & (pr.variant == "P_vs_N_rel_own") & pr.model.isin(pe)]
num("PrStrictRelLo", f"{q.auroc.min():.2f}")
num("PrStrictRelHi", f"{q.auroc.max():.2f}")
q = pr[(pr.setting == "t1_study_strict") & (pr.variant == "P_vs_N_degree_matched") & pr.model.isin(pe)]
num("PrStrictDegOnlyLo", f"{q.auroc.min():.2f}")
num("PrStrictDegOnlyHi", f"{q.auroc.max():.2f}")

ccl = cc[cc.model.isin(LEARNED)]
num("NocoocLearnedAllIncludeZero", "yes" if ((ccl.lo95 <= 0) & (ccl.hi95 >= 0)).all() else "no")

# summary macros for the abstract and introduction
mods = sorted({m for (_, m) in main.index if m not in ("random",)})
num("NMethods", str(len([m for m in mods if not m.endswith("-rs")])))
sig = pc.xs("same_minus_other", level="contrast")
sig = sig[(sig.lo95 > 0) & sig.index.isin(LEARNED + ["l3", "ra", "cn"])]
num("PSameSigLo", f"{sig.delta.min():.4f}")
num("PSameSigHi", f"{sig.delta.max():.4f}")
num("PSameSigN", str(len(sig)))
num("PSameSigNames", ", ".join(NAME[m] for m in sig.sort_values("delta").index))
nos = pc.xs("same_minus_other", level="contrast")
nos = nos[(nos.lo95 <= 0) & nos.index.isin(LEARNED)]
num("PSameNoGain", ", ".join(NAME[m] for m in nos.index) or "none")
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_det_degree_matched") & pr.model.isin(pe)]
num("PrAllMatchedAboveNames", "; ".join(f"{NAME[m]} on {HEAD[s]} ({r.auroc:.2f})" for (s, m), r in
                                        q.set_index(["setting", "model"]).iterrows() if r.lo > 0.5) or "none")
num("PrAllMatchedBelowNames", "; ".join(f"{NAME[m]} on {HEAD[s]} ({r.auroc:.2f})" for (s, m), r in
                                        q.set_index(["setting", "model"]).iterrows() if r.hi < 0.5) or "none")
q = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_within") & pr.model.isin(pe)]
num("PrAllWithinNotBelow", "; ".join(f"{NAME[m]} on {HEAD[s]}" for (s, m), r in
                                     q.set_index(["setting", "model"]).iterrows() if r.hi >= 0.5) or "none")

# ------------------------------------------------------------------------------------------ number macros
# random-search sizes and selections (results3/hpo_random/selected.json, copied to results3/analysis/hpo_selected.json)
hs = A3 / "hpo_selected.json"
if hs.exists():
    sel = json.load(open(hs))
    for k, v in sel.items():
        s_, m_ = k.split("|")
        tag = ("Strict" if s_ == "t1_study_strict" else "Ext") + m_.replace("RGCN", "Rgcn")
        num(f"RsN{tag}", str(v["n_configs"]))
        if v.get("best") and v.get("old_grid_valid_mrr") is not None:
            num(f"RsGain{tag}", f"{v['best']['valid_mrr'] - v['old_grid_valid_mrr']:+.4f}")
    lines = []
    for k, v in sel.items():
        s_, m_ = k.split("|")
        if not v.get("best"):
            continue
        c = v["best"]["config"]
        cfg = f"$d$={c['dim']}, lr={c['lr']:.2g}, {'NSSA' if c['loss'] == 'nssa' else 'default'}" + (
            f" ({c['nssa_margin']:g})" if c.get("nssa_margin") else "") + f", {c['num_negs']} neg., batch {c['batch']}"
        if c.get("rgcn_sampler"):
            cfg += f", {c['rgcn_sampler']}"
        ov = v.get("old_grid_valid_mrr")
        lines.append(f"{HEAD[s_]} & {m_.replace('RGCN', 'R-GCN')} & {v['n_configs']} & {v['best']['valid_mrr']:.4f} & "
                     f"{'--' if ov is None else f'{ov:.4f}'} & {{\\scriptsize {cfg}}} \\\\")
    write("app_hpo_random.tex", tab("llrrrp{6.2cm}", "Regime & Model & Configs & Best valid MRR & Grid valid MRR & "
                                                   "Selected configuration", lines))
for tag in ["StrictRotatE", "StrictTuckER", "StrictRgcn", "ExtRotatE", "ExtTuckER"]:
    NUM.setdefault(f"RsN{tag}", "\\textbf{[pending]}")
if not hs.exists():
    write("app_hpo_random.tex", "\\textbf{[pending: random search in progress]}\n")
cp_ = A3 / "compute.json"
if cp_.exists():
    cj = json.load(open(cp_))
    for k, nm in [("heuristics", "Heur"), ("embedding", "Emb"), ("rgcn", "Rgcn"), ("nbfnet", "Nbf"), ("ncn", "Ncn"),
                  ("gbm", "Gbm"), ("ultra_zeroshot", "UltraZs"), ("ultra_finetune", "UltraFt"),
                  ("random_search_refits", "Refit")]:
        num(f"Compute{nm}", f"{cj['cells'][k]['hours']:.1f}")
    num("ComputeCells", f"{cj['cells_total_hours']:.0f}")
    num("ComputeGrid", f"{cj['tuning']['grid_tuning']['hours']:.1f}")
    num("ComputeSearch", f"{cj['tuning']['random_search']['hours']:.0f}")
# macros that depend on runs still in the GPU queues get a visible placeholder until the runs finish
for k in ["RsRotateStrict", "RsRotateExt", "RsTuckerStrict", "RsTuckerExt", "RsRgcnStrict"]:
    NUM.setdefault(k, "\\textbf{[pending]}")
for s_, k in [("t1_pair", "Pair"), ("t1_study", "Study"), ("t1_study_strict", "Strict"), ("t1_temporal", "Temporal")]:
    if ("t1_study_strict", "nbfnet") in vs.index or (s_, "nbfnet") in vs.index:
        if (s_, "nbfnet") in vs.index:
            r = vs.loc[(s_, "nbfnet")]
            num(f"Nbf{k}", f"{r.mrr_delta:+.3f} {ci(r.mrr_lo95, r.mrr_hi95)} ({r.mrr_label})")
    NUM.setdefault(f"Nbf{k}", "\\textbf{[pending]}")
# ------------------------------------------------------------------------------------------ text assumptions
# The main text states some results in words; warn when the data no longer support them.
def check(cond, msg):
    if not cond:
        print("TEXT CHECK FAILED:", msg)


wl = vs[vs.index.get_level_values(0).isin(TRANSFER) & vs.index.get_level_values(1).isin(LEARNED)]
check((wl.mrr_label == "better").sum() == 0, "abstract: no learned method better on MRR")
eq = wl[wl.mrr_label == "equivalent"].index.tolist()
check(eq == [("t1_study_strict", "tucker")], f"TuckER on Strict is the only MRR equivalent: {eq}")
hb = wl[wl.h10_label == "better"].index.tolist()
check(hb == [("t1_external", "ultra-zeroshot")], f"Hits@10 better only ULTRA-0 on Out-of-time: {hb}")
he = sorted(wl[wl.h10_label == "equivalent"].index.tolist())
check(he == [("t1_temporal", "tucker"), ("t1_temporal", "ultra-finetune"), ("t1_temporal", "ultra-zeroshot")],
      f"Hits@10 equivalents on Temporal: {he}")
check(NUM["BetterAfterExt"] == "0", "no method better on studies published after the snapshot")
check(set(NUM["PSameSigNames"].split(", ")) == {"RA", "L3", "RotatE", "ULTRA (zero-shot)"},
      f"same-study gain set: {NUM['PSameSigNames']}")
check(set(NUM["PSameNoGain"].split(", ")) <= {"TuckER", "R-GCN"}, f"no same-study gain: {NUM['PSameNoGain']}")
check(float(NUM["NocoocMaxAbs"]) <= 0.004 and NUM["NocoocLearnedAllIncludeZero"] == "yes",
      f"co-occurrence removal small, learned intervals include 0: {NUM['NocoocMaxAbs']}")
check(int(NUM["VarNocohortGapNeg"]) >= int(NUM["VarNocohortGapN"]) - 2, "cohort removal widens most gaps")
check(NUM["PrAllWithinNotBelow"] == "R-GCN on Study-excl.", f"P/N not below 0.5: {NUM['PrAllWithinNotBelow']}")
check(all(x.split(" on ")[0] in ("CN", "RA", "L3") for x in NUM["PrAllMatchedBelowNames"].split("; ")),
      f"matched below 0.5 only CN/RA/L3: {NUM['PrAllMatchedBelowNames']}")
check(NUM["PoolTrainBetter"] == "0", "no method better with the reduced pool")
for k in ["BetterReplStrict", "BetterRerepStrict", "BetterRerepTemporal", "BetterTwoNewExt"]:
    check(NUM[k] == "0", f"{k} = {NUM[k]}")
check(NUM["ColdDisBetter"] == "1" and NUM["ColdDisUltraLabel"] == "better", "cold disease: only ULTRA-0 better")
ab = pr[(pr.setting == "t1_study_strict") & (pr.variant == "P_vs_N_det_degree_matched") & (pr.lo > 0.5)].model.tolist()
check(ab == ["gbm"], f"GBM is the only matched interval above 0.5 on Strict: {ab}")
qa = pr[pr.setting.isin(TRANSFER) & (pr.variant == "P_vs_N_det_degree_matched") & pr.model.isin(pe) & (pr.lo > 0.5)]
check(set(qa.setting) <= {"t1_temporal"} and (qa.auroc <= 0.56).all(),
      f"matched intervals above 0.5 only on Temporal and slight: {qa[['setting', 'model', 'auroc']].values.tolist()}")
check(float(NUM["VarNocohortGapHi"]) < 0, "cohort removal widens every learned gap")
check(float(NUM["PPermL3"]) < 0.05 and float(NUM["PPermULTRA"]) < 0.05 and float(NUM["PPermRotatE"]) >= 0.05,
      "permutation: L3 and ULTRA significant, RotatE not")
check((wl.mrr_mde80 > 0.005).mean() > 0.5, "most MDE80 exceed delta")
check(int(NUM["PairBetter"]) >= int(NUM["PairN"]) - 2, "most learned methods beat the prior on Pair")
DIG = str.maketrans({"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
                     "7": "Seven", "8": "Eight", "9": "Nine"})
clean = {k.replace("H10", "Hits").translate(DIG): v for k, v in NUM.items()}
assert all(k.isalpha() for k in clean), [k for k in clean if not k.isalpha()]
lines = [f"\\newcommand{{\\n{k}}}{{{tex_minus(v)}}}" for k, v in sorted(clean.items())]
(OUT / "numbers.tex").write_text("% generated by the table-generation script in code/\n" + "\n".join(lines) + "\n")
print("numbers:", len(NUM))
