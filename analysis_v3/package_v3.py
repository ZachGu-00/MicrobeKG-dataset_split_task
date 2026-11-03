"""Build deliverable_v3/: compiled paper, LaTeX source zip, release zip and analysis-code zip.

The release zip has the root folder v3/ (README commands run from inside it).  Caches are excluded."""
import shutil
import zipfile
from pathlib import Path

R = Path("/home/user/ClawsGO/MicrobeKG")
OUT = R / "deliverable_v3"
OUT.mkdir(exist_ok=True)
SKIP = ("__pycache__", ".pytest_cache", ".egg-info")


def zipdir(zpath, items):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in items:
            src = Path(src)
            files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
            for p in files:
                if any(s in str(p) for s in SKIP):
                    continue
                z.write(p, arc if src.is_file() else str(Path(arc) / p.relative_to(src)))
    print(zpath.name, zpath.stat().st_size)


M = R / "manuscript_v3"
shutil.copy2(M / "main.pdf", OUT / "MicrobeKG-Bench_NeurIPS2026_v3.pdf")
zipdir(OUT / "MicrobeKG-Bench_latex_source_v3.zip",
       [(M / f, f"latex/{f}") for f in ["main.tex", "appendix.tex", "related_ext.tex", "checklist_filled.tex",
                                         "refs.bib", "main.bbl", "neurips_2026.sty"]]
       + [(M / "tables", "latex/tables"), (M / "figures", "latex/figures")])
zipdir(OUT / "MicrobeKG-Bench_release_v3.zip", [(R / "release/v3", "v3")])
code = ["analysis_v3.py", "build_variants_v3.py", "gbm_v3.py",
        "hpo_random_v3.py", "jobs_ctx_v3.py", "make_tables_v3.py", "nbfnet_v3.py", "probe_v3.py", "queue_v3.py",
        "release_reference_v3.py", "replication_rate_v3.py", "rereport_v3.py", "rescore_v3.py", "sched_one_v3.py",
        "sched_variants_v3.py", "sibling_audit_v3.py", "ultra_v3.py", "mk3_cell.py", "mkbench_build_v2.py",
        "analysis_v2.py", "package_v3.py"]
zipdir(OUT / "MicrobeKG-Bench_analysis_code_v3.zip",
       [(R / "code" / f, f"code/{f}") for f in code if (R / "code" / f).exists()]
       + [(R / "code/figstyle/mkstyle.py", "code/figstyle/mkstyle.py"),
          (R / "work/baselines_v3/ncn_run.py", "code/baselines/ncn_run.py"),
          (R / "audit_hmdad/kgnmda", "code/kgnmda")])
