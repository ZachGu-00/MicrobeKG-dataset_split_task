#!/usr/bin/env python3
"""Map the audit's HMDAD microbes/diseases onto KGNMDA's shipped knowledge-graph entity ids.

Microbes: audit names -> NCBI taxids (E-utilities, see map_taxa.py) -> `Microbe::TAXONOMY:<taxid>` in entity2id.txt.
Diseases: the KGNMDA repo stores 39 HMDAD diseases as MeSH ids or free-text names. They are matched to the audit's
          disease names by exact normalised name where possible and otherwise by the Jaccard overlap of their
          associated microbe sets (Hungarian assignment).
Validation: the repo ships 449 positive (disease, microbe) pairs. The share of audit pairs reproduced exactly by the
          mapping is reported; a wrong mapping cannot reproduce hundreds of pairs by chance.
Output: hmdad_entity_map.json (aligned with the audit's sorted microbe / disease lists).
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

B = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/mkg/baselines_v3')
D = Path('/root/autodl-tmp/mkg/vendor/KGNMDA/raw_data/mdkg_hmdad')

ent = pd.read_csv(D / 'entity2id.txt', sep='\t', skiprows=1, header=None, names=['name', 'id'], dtype=str)
assert (ent.id.astype(int).values == np.arange(len(ent))).all(), 'entity id must equal row index'
n_ent = len(ent)
id2name = dict(zip(ent.id.astype(int), ent.name))
kg_tax = {n.split('TAXONOMY:')[1]: int(i) for n, i in zip(ent.name, ent.id) if n.startswith('Microbe::TAXONOMY:')}

ex = pd.read_csv(D / 'approved_example.txt', sep='\t', header=None, names=['d', 'm', 'y'], dtype=int)
repo_pos = ex[ex.y == 1]
repo_pairs = set(zip(repo_pos.d, repo_pos.m))
repo_dis = sorted(repo_pos.d.unique())
repo_dis_micro = {d: set(g.m) for d, g in repo_pos.groupby('d')}

audit = pd.read_csv(B / 'hmdad_pairs_audit.csv')
a_micro = sorted(audit.microbe.unique())
a_dis = sorted(audit.disease.unique())
cand = pd.read_csv(B / 'microbe_taxid_candidates.csv', dtype=str).fillna('').set_index('microbe')

# ---- microbes: candidate KG entities by name
micro_cands = {}
for m in a_micro:
    ids = [kg_tax[t] for t in cand.loc[m, 'in_kg'].split(';') if t] if m in cand.index else []
    micro_cands[m] = ids
unique_m = {m: c[0] for m, c in micro_cands.items() if len(c) == 1}

# ---- diseases: name match, then Jaccard on uniquely mapped microbes
norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
a_dis_micro = {d: {unique_m[m] for m in g.microbe if m in unique_m} for d, g in audit.groupby('disease')}
S = np.zeros((len(a_dis), len(repo_dis)))
for i, d in enumerate(a_dis):
    for j, r in enumerate(repo_dis):
        A, R = a_dis_micro[d], repo_dis_micro[r]
        jac = len(A & R) / max(1, len(A | R))
        rn = id2name[r].split('::', 1)[1]
        S[i, j] = jac + (1.0 if norm(rn) == norm(d) else 0.0)
ri, ci = linear_sum_assignment(-S)
dis_map = {a_dis[i]: int(repo_dis[j]) for i, j in zip(ri, ci)}
dis_score = {a_dis[i]: float(S[i, j]) for i, j in zip(ri, ci)}

# ---- resolve ambiguous / missing microbes against the repo's own pair list
final, how = {}, {}
for m in a_micro:
    c = micro_cands[m]
    dset = set(audit[audit.microbe == m].disease)
    if len(c) == 1:
        final[m], how[m] = c[0], 'name_unique'
    elif len(c) > 1:
        score = [sum((dis_map[d], x) in repo_pairs for d in dset) for x in c]
        final[m], how[m] = c[int(np.argmax(score))], 'name_ambiguous_resolved_by_pairs'
    else:
        final[m], how[m] = None, 'unmapped'

m_dis = {m: {dis_map[d] for d in g.disease} for m, g in audit.groupby('microbe')}


def hits(m, eid):
    """number of the audit pairs of microbe m that the repo ships for KG entity eid"""
    return sum((d, eid) in repo_pairs for d in m_dis[m])


# (1) independent check of the NCBI name mapping, before any repair: all pairs of a name-mapped microbe should be shipped by the repo
stage1 = {m: (hits(m, final[m]), len(m_dis[m])) for m in a_micro if final[m] is not None}
name_stage = {'microbes_name_mapped': len(stage1),
              'microbes_all_pairs_found_in_repo': sum(h == n for h, n in stage1.values()),
              'pairs_name_mapped': sum(n for _, n in stage1.values()), 'pairs_found_in_repo': sum(h for h, _ in stage1.values())}

# (2) collisions: several audit microbes on one entity -> the one with most repo-confirmed pairs keeps it, the others are demoted
owners = {}
for m in a_micro:
    if final[m] is not None:
        owners.setdefault(final[m], []).append(m)
for eid, ms in owners.items():
    if len(ms) > 1:
        ms = sorted(ms, key=lambda m: (-hits(m, eid), -len(m_dis[m]), m))
        for m in ms[1:]:
            final[m], how[m] = None, 'collision_demoted'
# (3) a mapped microbe whose pairs the repo does not ship is a wrong name match
for m in a_micro:
    if final[m] is not None and hits(m, final[m]) < len(m_dis[m]) and how[m] != 'name_ambiguous_resolved_by_pairs':
        final[m], how[m] = None, 'name_mismatch_demoted'
# (4) recover unresolved microbes by an unambiguous disease-profile match among still unclaimed repo microbes
used = {v for v in final.values() if v is not None}
repo_micro_dis = {}
for d, m in repo_pairs:
    repo_micro_dis.setdefault(m, set()).add(d)
free = {m: ds for m, ds in repo_micro_dis.items() if m not in used}
for m in a_micro:
    if final[m] is None:
        prof = m_dis[m]
        hit = [x for x, ds in free.items() if ds == prof]
        if len(hit) == 1 and len(prof) >= 2:   # singleton profiles are not identifiable
            final[m], how[m] = hit[0], 'profile_match'
            free.pop(hit[0])

# ---- validation (final mapping; (1) above is the independent number)
mapped_pairs = {(dis_map[r.disease], final[r.microbe]) for r in audit.itertuples() if final[r.microbe] is not None}
n_audit_pairs = len(audit)
n_in_repo = len(mapped_pairs & repo_pairs)
n_unmapped_micro = sum(v is None for v in final.values())
n_unmapped_pairs = int(sum(final[r.microbe] is None for r in audit.itertuples()))
extra = [m for m in a_micro if final[m] is None]
extra_ids = {m: n_ent + k for k, m in enumerate(extra)}   # isolated pseudo-entities appended after the KG vocabulary
micro_eid = [final[m] if final[m] is not None else extra_ids[m] for m in a_micro]
dis_eid = [dis_map[d] for d in a_dis]
assert len(set(micro_eid)) == len(micro_eid), 'two audit microbes share an entity id'
assert len(set(dis_eid)) == len(dis_eid)

out = {'n_entities_kg': n_ent, 'n_entities_total': n_ent + len(extra),
       'microbes': a_micro, 'micro_eid': micro_eid, 'micro_how': [how[m] for m in a_micro],
       'diseases': a_dis, 'dis_eid': dis_eid, 'dis_name_in_kg': [id2name[x] for x in dis_eid], 'dis_match_score': [dis_score[d] for d in a_dis],
       'unmapped_microbes': extra,
       'validation': {'audit_pairs': n_audit_pairs, 'repo_positive_pairs': len(repo_pairs),
                      'audit_pairs_with_mapped_microbe': n_audit_pairs - n_unmapped_pairs,
                      'mapped_pairs_found_in_repo_examples': n_in_repo,
                      'repo_pairs_not_explained': len(repo_pairs - mapped_pairs),
                      'name_stage_before_repair': name_stage,
                      'unmapped_microbes': n_unmapped_micro, 'unmapped_microbe_pairs': n_unmapped_pairs,
                      'how_counts': pd.Series(list(how.values())).value_counts().to_dict()}}
(B / 'hmdad_entity_map.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out['validation'], indent=1))
print('disease matches (audit -> KG name, score):')
for d, x, s in zip(a_dis, out['dis_name_in_kg'], out['dis_match_score']):
    print(f'  {d:50s} -> {x:55s} {s:.2f}')
print('unmapped microbes:', extra)
