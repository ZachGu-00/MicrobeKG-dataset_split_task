# Sources, attribution, and licensing observations

Counts refer to microbekg-audited and were independently recomputed from the exported records.
They count assertion rows carrying each source label; multi-source rows contribute to more than one count.
The 19 labels consist of 18 upstream labels and one derived conflict label.

This is a source-terms record, not an unrestricted public redistribution license.
`verified` / `verified_terms` indicate a located statement, not clearance for every possible use.
`pending` means the applicable permission has not been established, not that reuse is forbidden.
`partial` means the located statement does not yet establish the exact downloaded payload's terms.
`restricted` / `restricted_pending_snapshot` preserve known qualifications.
Publication licensing and database or supplementary-table licensing may differ.

The earlier source review was dated 2026-09-18. Only HMDB and CTD were refreshed on 2026-09-29
for this packaging task; dates below do not imply a new review of the other providers.
See [source_terms.json](source_terms.json) for the machine-readable record and [LICENSE](LICENSE) for scope.

| Source label | Labelled assertions | Status | Terms observation | Last checked | Evidence |
|---|---:|---|---|---|---|
| AGORA2 | 1,496,721 | partial | MIT on repository; downloaded SBML scope needs confirmation | 2026-09-18 | [Provider / publication](https://github.com/VirtualMetabolicHuman/AGORA2) |
| BugSigDB | 15,197 | verified | ODC-BY-1.0 | 2026-09-18 | [Provider / publication](https://bugsigdb.org/Project:About) |
| CTD | 124,973 | restricted | Non-commercial access; commercial download license | 2026-09-29 | [Provider / publication](https://academic.oup.com/nar/article/53/D1/D1328/7816860) |
| Disbiome | 6,715 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://disbiome.ugent.be/) |
| GMMAD | 1,046,640 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10464125/) |
| GMrepo_v3 | 19,849 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://gmrepo.humangut.info/) |
| HMDAD | 407 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](http://www.cuilab.cn/hmdad) |
| HMDB | 850,148 | restricted_pending_snapshot | Official terms require explicit permission and acknowledgment for commercial use and redistribution. | 2026-09-29 | [Provider / publication](https://www.hmdb.ca/downloads) |
| Lit44 | 1,941 | pending | Per-article extracted factual assertions; heterogeneous terms | 2026-09-18 | Study identifiers in each record's evidence |
| MeSH | 8,076 | verified_terms | NLM MeSH Terms and Conditions | 2026-09-18 | [Provider / publication](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html) |
| NCBI_Taxonomy | 58,610 | verified_terms | NCBI government-information policy, subject to third-party notices | 2026-09-18 | [Provider / publication](https://www.ncbi.nlm.nih.gov/home/about/policies/) |
| NJC19 | 5,042 | partial | Dryad dataset terms; snapshot license confirmation pending | 2026-09-18 | [Provider / publication](https://datadryad.org/dataset/doi:10.5061/dryad.dr7sqv9v8) |
| Peryton | 3,439 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://dianalab.e-ce.uth.gr/peryton/) |
| Piccinno_2025 | 191 | pending | Source/supplement-specific terms to confirm | 2026-09-18 | [Provider / publication](https://doi.org/10.1038/s41591-025-03693-9) |
| Thomas_2019 | 52 | pending | Source/supplement-specific terms to confirm | 2026-09-18 | [Provider / publication](https://doi.org/10.1038/s41591-019-0405-7) |
| cross_source_conflict | 1,054 | derived | Derived; preserve all contributing-source restrictions | 2026-09-18 | [Provider / publication](https://github.com/ZachGu-00/MicrobeKG-dataset_split_task/blob/2324306309e8851596b443e2711fef4b1bf41d66/scripts/step_merge_kg.py) |
| gutMDisorder_v2 | 2,041 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://bio-computing.hrbmu.edu.cn/gutMDisorder/) |
| gutMGene | 4,771 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://bio-computing.hrbmu.edu.cn/gutmgene/) |
| mBodyMap | 8,252 | pending | No explicit data license established | 2026-09-18 | [Provider / publication](https://mbodymap.microbiome.cloud/) |

## Source-specific notes

- **AGORA2**: Repository redirects model downloads to VMH; preserve model attribution and confirm SBML terms.
- **BugSigDB**: Data licensing section explicitly covers database reuse; preserve attribution and notices.
- **CTD**: The authors describe free non-commercial availability and a commercial download license. This does not establish unrestricted bulk redistribution of the derived graph.
- **Disbiome**: Download access and article availability do not establish redistribution rights.
- **GMMAD**: Database paper located; exact downloaded-table terms still required.
- **GMrepo_v3**: Confirm terms for the v3 phenotype and abundance exports used here.
- **HMDAD**: Confirm original association table license.
- **HMDB**: Commercial restriction verified in current official page text. Exact source snapshot and applicable redistribution permission remain to be confirmed.
- **Lit44**: Historical source label. Retained assertion evidence contains study identifiers; exact article/supplement terms require per-study review. Full texts are not included.
- **MeSH**: Acknowledge NLM; no endorsement; identify frozen version (local source desc2026.xml.gz).
- **NCBI_Taxonomy**: Acknowledge NCBI and retain snapshot/version information; not a blanket license for other NCBI-hosted resources.
- **NJC19**: Exact upstream dataset and NJC19 archive located; retain Lim et al. 2020 citation.
- **Peryton**: Confirm downloadable associations terms separately from paper license.
- **Piccinno_2025**: Confirm license and third-party notices for the exact MaAsLin2/phylogenetic/meta-analysis tables.
- **Thomas_2019**: Confirm exact supplement identity and reuse terms; not inferred from another article.
- **cross_source_conflict**: Conflict mining is original processing; underlying evidence remains source-governed.
- **gutMDisorder_v2**: Freely accessible resource does not establish an unrestricted data license.
- **gutMGene**: Confirm terms for the three downloaded interaction tables.
- **mBodyMap**: Confirm terms for phenotype/body-site/species exports.

## Attribution

Acknowledge MicrobeKG and the original providers whose records are used. Follow each provider's
requested citation and preserve the source/evidence fields. MeSH and NCBI inclusion does not
imply endorsement by NLM, NCBI, or the United States government. This package does not change
upstream licensing conditions or certify that all third-party permissions have been obtained.
