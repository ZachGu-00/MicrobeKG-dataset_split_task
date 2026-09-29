"""Evidence-only tables, paired bootstrap and source degree audits; no model fitting."""
import argparse
from collections import Counter
import hashlib
import numpy as np
import pandas as pd

from common import ROOT, OUT, PLAN, read, write_json, sha256, folder, verified
from core import load_case
from prepare import CORE, SEEDS

REPORT=OUT/'analysis'
HIST=ROOT/'autodl/supplement_results/analysis'


def load_raw(path):
    result=read(ROOT/path)
    raw_path=(ROOT/path).with_suffix('.npz')
    assert result['raw_sha256']==sha256(raw_path), str(raw_path)
    with np.load(raw_path,allow_pickle=False) as z: return dict(z)


def records():
    data=pd.read_csv(HIST/'all_observed_results.csv').to_dict('records')
    base={('shuffle' if r['group']=='shuffle' else 'main',r['setting'],r['model'],int(r['seed'])):r
          for r in data if r['group']!='diagnostic'}
    # A corrected cell never silently falls back to its superseded old configuration.
    jobs=read(PLAN)['jobs']; pending=[]
    for job in jobs:
        if job['kind'] in ['hpo','matched']: continue
        group='shuffle' if job['kind']=='shuffle' else 'main'
        key=(group,job['setting'],job['model'],job['seed'])
        base.pop(key,None)
        if verified(job):
            base[key]=dict(group=group,setting=job['setting'],model=job['model'],seed=job['seed'],
                          result_file=(folder(job)/'result.json').relative_to(ROOT).as_posix(),origin='revision')
        else:
            receipt=read(folder(job)/'receipt.json') if (folder(job)/'receipt.json').exists() else {}
            pending.append(dict(id=job['id'],setting=job['setting'],model=job['model'],seed=job['seed'],
                                status=receipt.get('status','not_run'),error=receipt.get('error')))
    return base,pending


def tables(base,pending):
    rows=[]
    for (group,setting,model,seed),record in base.items():
        raw=load_raw(record['result_file']); sides={}
        for side in ['head','tail']:
            if side+'_ranks' in raw: sides[side+'_mrr']=float(np.mean(1/raw[side+'_ranks']))
        result=read(ROOT/record['result_file'])
        rows.append(dict(group=group,setting=setting,model=model,seed=seed,
            result_file=record['result_file'],mrr=np.mean(list(sides.values())) if sides else np.nan,**sides,
            **{k+'_auroc':v['auroc'] for k,v in result.get('discrimination',{}).items()}))
    df=pd.DataFrame(rows); df.to_csv(REPORT/'observed_per_seed.csv',index=False)
    metrics=[c for c in df if c.endswith(('_mrr','_auroc')) or c=='mrr']
    summary=[]
    expected={(('shuffle' if j['kind']=='shuffle' else 'main'),j['setting'],j['model'])
              for j in read(PLAN)['jobs'] if j['kind'] not in ['hpo','matched']}
    expected|=set(map(tuple,df[['group','setting','model']].drop_duplicates().values))
    for group,setting,model in sorted(expected):
        d=df[(df.group==group)&(df.setting==setting)&(df.model==model)]
        row=dict(group=group,setting=setting,model=model,n_seeds=len(d),status='complete' if len(d)==3 else 'pending')
        for metric in metrics:
            # Headline mean and SD require all three seeds with the current protocol.
            row[metric+'_mean']=d[metric].mean() if len(d)==3 else np.nan
            row[metric+'_sd']=d[metric].std() if len(d)==3 else np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(REPORT/'main_head_tail.csv',index=False)
    write_json(REPORT/'pending_jobs.json',pending)
    return df,pd.DataFrame(summary)


def matched_tables():
    old=pd.read_csv(HIST/'matched_all_observed.csv')
    mapping={(r.setting,r.model,int(r.seed)):r.result_file for r in old.itertuples()}
    expected=set(mapping)
    for job in read(PLAN)['jobs']:
        if job['kind']!='matched': continue
        key=(job['setting'],job['model'],job['seed']); expected.add(key); mapping.pop(key,None)
        if verified(job): mapping[key]=(folder(job)/'result.json').relative_to(ROOT).as_posix()
    rows=[]
    for (setting,model,seed),path in mapping.items():
        result=read(ROOT/path); load_raw(path)
        for control,metrics in result['metrics'].items():
            rows.append(dict(setting=setting,model=model,seed=seed,control=control,result_file=path,
                             **{k:metrics[k] for k in ['auroc','auprc','floor','n_pos','n_neg']}))
    df=pd.DataFrame(rows);df.to_csv(REPORT/'matched_per_seed.csv',index=False)
    summary=[]
    for setting,model in sorted({(s,m) for s,m,_ in expected}):
        d=df[(df.setting==setting)&(df.model==model)]
        for control in ['original','recorded_degree_matched','original_controls_on_matched_positive_subset','unknown_fallback_only']:
            v=d[d.control==control]; n=len(v)
            summary.append(dict(setting=setting,model=model,control=control,n_seeds=n,status='complete' if n==3 else 'pending_or_unavailable',
                **{metric+'_'+stat:(getattr(v[metric],stat)() if n==3 else np.nan) for metric in ['auroc','auprc'] for stat in ['mean','std']}))
    pd.DataFrame(summary).to_csv(REPORT/'matched_mean_std.csv',index=False)


def paired_vectors(a,b,side='both'):
    # Align identities, not row positions; require identical queries and candidate pools.
    ka=[tuple(x) for x in a['ranking_triples']]; kb=[tuple(x) for x in b['ranking_triples']]
    assert len(set(ka))==len(ka) and len(set(kb))==len(kb), 'duplicate scoring triples'
    assert set(ka)==set(kb), 'paired test queries differ'
    lookup={k:i for i,k in enumerate(kb)}; order=np.array([lookup[k] for k in ka])
    sides=[s for s in ['head','tail'] if s+'_ranks' in a] if side=='both' else [side]
    delta=[]
    for s in sides:
        pa,pb=a[s+'_pool_size'],b[s+'_pool_size']
        np.testing.assert_array_equal(pa,pb[order] if np.ndim(pb) else pb)
        delta.append(1/a[s+'_ranks']-1/b[s+'_ranks'][order])
    return np.array(ka)[:,[0,2]],np.mean(delta,axis=0)


def cluster_bootstrap(pairs,delta,n_boot,seed):
    _,group=np.unique(pairs,axis=0,return_inverse=True)
    n=int(group.max())+1
    sums=np.bincount(group,weights=delta); sizes=np.bincount(group)
    rng=np.random.default_rng(seed); samples=[]
    for start in range(0,n_boot,32):
        ids=rng.integers(n,size=(min(32,n_boot-start),n))
        samples.extend(sums[ids].sum(1)/sizes[ids].sum(1))
    return np.asarray(samples),n


def weighted_auc(pos,neg,pw,nw):
    order=np.argsort(neg,kind='stable'); sorted_neg=neg[order]
    c=np.r_[0.,np.cumsum(nw[order])]
    lower=c[np.searchsorted(sorted_neg,pos,'left')]; upper=c[np.searchsorted(sorted_neg,pos,'right')]
    return float(np.dot(pw,(lower+upper)/2)/(pw.sum()*nw.sum()))


def control_bootstrap(source,match,n_boot,seed):
    label='association_without_recorded_therapy'
    pos,neg=source[label+'_pos_scores'],source[label+'_neg_scores']
    for side,key in [('positive','pos'),('negative','neg')]:
        np.testing.assert_array_equal(source[label+'_'+key+'_ids'],match['original_'+side+'_ids'])
    pi,ni=match['positive_indices'],match['negative_indices']
    assert len(np.unique(pi))==len(pi) and len(np.unique(ni))==len(ni)
    pos=pos[pi]; m=len(pi); other=np.setdiff1d(np.arange(len(neg)),ni)
    ones=np.ones(m); base=weighted_auc(pos,neg[ni],ones,ones)-weighted_auc(pos,neg,ones,np.ones(len(neg)))
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(n_boot):
        weights=np.bincount(rng.integers(m,size=m),minlength=m)
        nw=np.zeros(len(neg)); nw[ni]=weights
        if len(other): nw[other]=np.bincount(rng.integers(len(other),size=len(other)),minlength=len(other))
        samples.append(weighted_auc(pos,neg[ni],weights,weights)-weighted_auc(pos,neg,weights,nw))
    return base,np.asarray(samples),m


def paired_analyses(base,n_boot):
    rows=[]; pending=[]; groups={}
    def save(family,setting,a,b,side,seed,point,boots,n,method):
        key=(family,setting,a,b,side)
        groups.setdefault(key,[]).append((seed,point,boots,n,method))
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append(dict(finding=family,setting=setting,A=a,B=b,side=side,seed=seed,
                         delta=point,ci95_low=lo,ci95_high=hi,n_clusters=n,n_boot=n_boot,method=method))
    def comparison(family,setting,a,b,group_a='main',group_b='main',side='both'):
        for seed in SEEDS:
            ka,kb=(group_a,setting,a,seed),(group_b,setting,b,seed)
            if ka not in base or kb not in base:
                pending.append(dict(finding=family,setting=setting,A=a,B=b,side=side,seed=seed)); continue
            pairs,delta=paired_vectors(load_raw(base[ka]['result_file']),load_raw(base[kb]['result_file']),side)
            rngseed=int(hashlib.sha256(str((family,setting,a,b,side,seed)).encode()).hexdigest()[:8],16)
            boots,n=cluster_bootstrap(pairs,delta,n_boot,rngseed)
            save(family,setting,a+(' [shuffle]' if group_a=='shuffle' else ''),b+(' [shuffle]' if group_b=='shuffle' else ''),
                 side,seed,float(delta.mean()),boots,n,'paired typed-pair cluster bootstrap; head/tail and relations kept together')
    for setting in ['t1_pair','t1_loso_peryton','t1_loso_gutmdisorder_v2','t1_loso_hmdad']:
        comparison('F1_source_transfer',setting,'RotatE','Popularity')
    for setting in CORE[:4]+['t1_loso_peryton','t1_loso_gutmdisorder_v2','t1_loso_hmdad']:
        for model in ['ComplEx','DistMult']:
            comparison('D_bilinear_vs_random',setting,model,'Random')
    for seed in SEEDS:
        key=('main','t4_b','DistMult',seed)
        with np.load(ROOT/f'supplement_data/matched/t4_b/seed_{seed}.npz') as z: match=dict(z)
        point,boots,n=control_bootstrap(load_raw(base[key]['result_file']),match,n_boot,seed)
        save('F2_control_selection','t4_b','recorded_degree_matched','original_controls_same_positives','AUROC',seed,point,boots,n,
             'paired matched blocks; shared original controls carry same weights; residual controls independently resampled')
    for setting,models in [('t1_pair',['TransE','DistMult','ComplEx','RotatE','TuckER','RGCN']),('t4_a',['DistMult','RGCN'])]:
        for model in models:
            comparison('F3_target_shuffle',setting,model,model,group_b='shuffle')
            comparison('F3_shuffle_residual',setting,model,'Popularity',group_a='shuffle')
    for setting in CORE[:4]+['t2_a']:
        for model in ['ULTRA','KG-ICL']:
            for comparator in ['RotatE','TuckER']:
                comparison('F4_modern_models',setting,model,comparator)
                if setting in ['t1_cold_microbe','t1_cold_disease']:
                    comparison('F4_cold_endpoint',setting,model,comparator,
                               side='head' if setting=='t1_cold_microbe' else 'tail')
    for key,values in groups.items():
        if {v[0] for v in values}!=set(SEEDS): continue
        family,setting,a,b,side=key
        boots=np.mean([v[2] for v in values],axis=0); lo,hi=np.quantile(boots,[.025,.975])
        rows.append(dict(finding=family,setting=setting,A=a,B=b,side=side,seed='mean_3_fixed_splits',
            delta=np.mean([v[1] for v in values]),ci95_low=lo,ci95_high=hi,n_clusters=sum(v[3] for v in values),n_boot=n_boot,
            method='mean of independently paired within-split bootstraps; conditional on these fixed splits, not seed-generalization CI'))
    pd.DataFrame(rows).to_csv(REPORT/'paired_bootstrap.csv',index=False)
    write_json(REPORT/'paired_pending.json',pending)
    return rows


def source_audit(base):
    sources=read(ROOT/'supplement_data/t1_pair_sources.json')['pair_sources']
    total=Counter(s for ss in sources.values() for s in ss)
    exclusive=Counter(ss[0] for ss in sources.values() if len(ss)==1)
    pd.DataFrame([dict(source=s,total_pairs=total[s],exclusive_pairs=exclusive[s],shared_pairs=total[s]-exclusive[s])
                  for s in sorted(total)]).to_csv(REPORT/'all_source_pair_counts.csv',index=False)
    summaries=[]; pair_rows=[]; distributions=[]
    for source in ['Peryton','gutMDisorder_v2','HMDAD']:
        setting='t1_loso_'+source.lower()
        for seed in SEEDS:
            c=load_case(ROOT/'supplement_data/loso',setting,seed)
            train=c['train']; degree=np.bincount(train[:,[0,2]].ravel(),minlength=len(c['seen']))
            raw=load_raw(base['main',setting,'Popularity',seed]['result_file'])
            test=raw['ranking_triples']; pair=test[:,[0,2]]
            target=train[np.isin(train[:,1],[c['relations'].index(r) for r in c['meta']['scoring_relations']])]
            td=np.bincount(target[:,[0,2]].ravel(),minlength=len(c['seen']))
            for side,col,typ in [('head',0,'microbe'),('tail',2,'disease')]:
                values=degree[test[:,col]]
                pool=np.flatnonzero(c['eval_seen'] & (c['entities'].node_type.to_numpy()==typ))
                threshold=np.quantile(degree[pool],.9)
                row=dict(source=source,seed=seed,side=side,exclusive_pairs=exclusive[source],shared_pairs=total[source]-exclusive[source],
                    evaluated_pairs=len(np.unique(pair,axis=0)),n_queries=len(pair),n_candidates=len(pool),
                    top_decile_degree_threshold=float(threshold),test_fraction_in_top_decile=float(np.mean(values>=threshold)),
                    popularity_mrr=float(np.mean(1/raw[side+'_ranks'])))
                row.update({f'degree_q{q}':float(np.percentile(values,q)) for q in [0,25,50,75,90,99,100]})
                summaries.append(row)
                for population,z in [('heldout_query_targets',values),('train_visible_candidates',degree[pool])]:
                    bins=np.floor(np.log2(np.maximum(z,1))).astype(int)
                    for b,n in sorted(Counter(bins).items()):
                        distributions.append(dict(source=source,seed=seed,side=side,population=population,log2_bin=int(b),count=n,fraction=n/len(z)))
            for i,(h,t) in enumerate(pair):
                pair_rows.append(dict(source=source,seed=seed,head=int(h),tail=int(t),
                    head_degree=int(degree[h]),tail_degree=int(degree[t]),head_target_degree=int(td[h]),tail_target_degree=int(td[t]),
                    head_rr=float(1/raw['head_ranks'][i]),tail_rr=float(1/raw['tail_ranks'][i]),
                    hbin=int(np.floor(np.log2(max(degree[h],1)))),tbin=int(np.floor(np.log2(max(degree[t],1))))))
    pd.DataFrame(summaries).to_csv(REPORT/'loso_degree_summary.csv',index=False)
    pd.DataFrame(distributions).to_csv(REPORT/'loso_degree_distribution.csv',index=False)
    d=pd.DataFrame(pair_rows).groupby(['source','seed','head','tail'],as_index=False).mean()
    d.to_csv(REPORT/'loso_pair_degrees.csv',index=False)
    standard=[]
    for seed in SEEDS:
        s=d[d.seed==seed]; bins=s.groupby(['source','hbin','tbin']).size()
        shared=bins.reset_index().groupby(['hbin','tbin']).size(); common=set(shared[shared==3].index)
        kept=s[[tuple(x) in common for x in s[['hbin','tbin']].values]]
        weights=kept.groupby(['hbin','tbin']).size(); weights=weights/weights.sum()
        for source,part in s.groupby('source'):
            p=kept[kept.source==source]; means=p.groupby(['hbin','tbin'])[['head_rr','tail_rr']].mean()
            standard.append(dict(source=source,seed=seed,common_bins=len(common),coverage=len(p)/len(part),
                raw_pair_mrr=float(part[['head_rr','tail_rr']].values.mean()),
                common_support_unweighted_mrr=float(p[['head_rr','tail_rr']].values.mean()),
                common_degree_standardized_mrr=float(means.mean(axis=1).mul(weights).sum()),
                interpretation='descriptive common-support reweighting; source queries differ; no paired or causal claim'))
    pd.DataFrame(standard).to_csv(REPORT/'peryton_common_degree_standardization.csv',index=False)


def narrative(summary,paired,pending):
    lines=['# 补充实验结果（由原始输出生成）','',
           f'尚缺 {len(pending)} 个训练/基础模型结果；主表仅在同一配置的 3 个 seed 齐全时输出均值。',
           '旧结果完整保留；修正后的双线性模型不与旧配置混算。数值为 MRR 或指定的 AUROC 差值。','',
           '## 双向检索主表','', '| setting | model | n | head MRR | tail MRR | mean MRR |','|---|---|---:|---:|---:|---:|']
    fmt=lambda row,m: 'pending / N.A.' if pd.isna(row.get(m+'_mean',np.nan)) else f"{row[m+'_mean']:.4f} ± {row[m+'_sd']:.4f}"
    for r in summary[summary.group=='main'].to_dict('records'):
        lines.append(f"| {r['setting']} | {r['model']} | {r['n_seeds']} | {fmt(r,'head_mrr')} | {fmt(r,'tail_mrr')} | {fmt(r,'mrr')} |")
    lines+=['','## 四条发现的配对证据','',
            '下表是 A−B 的三固定划分平均差及条件 bootstrap 95% CI。CI 未作多重比较校正；全比较均报告，不按显著性筛选。',
            'F1 比较每个来源内部 RotatE 与 Popularity 的差，不将不相交的 LOSO 与 random 测试集伪配对。',
            'F2 固定相同正例子集，比较度匹配记录对照和原对照；未知 pair 回退单独保留。',
            'F3 分开报告目标边打乱效应与打乱后相对 Popularity 的残余；保留辅助边，不能据此认定生物机制。',
            'F4 比较两个基础模型与预先指定的 RotatE/TuckER；cold-M 重点看 head，cold-D 重点看 tail。','',
            '| finding | setting | A | B | side | delta [95% CI] |','|---|---|---|---|---|---|']
    for r in paired:
        if r['seed']=='mean_3_fixed_splits':
            lines.append(f"| {r['finding']} | {r['setting']} | {r['A']} | {r['B']} | {r['side']} | {r['delta']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] |")
    degrees=pd.read_csv(REPORT/'loso_degree_summary.csv')
    peryton=degrees[degrees.source=='Peryton']
    standardized=pd.read_csv(REPORT/'peryton_common_degree_standardization.csv')
    pp=standardized[standardized.source=='Peryton']
    lines+=['','## 解释与边界','',
            'ComplEx / DistMult 的验证集控制实验已发现配置退化：初始化尺度、关系正则与 NSSA margin 会改变学习结果。诊断报告随包附带；正式测试结论以本表完成后的结果为准。',
            'Peryton 的 Popularity 实现按训练图端点度数乘积评分；固定查询后候选排序只取决于候选度数。LOSO 的度分布、候选覆盖和共同度区间标准化见同目录 CSV。该分析检验基线为何占优的分布解释，不证明疾病机制或单一来源的因果效应。',
            f"Peryton 有 {int(peryton.exclusive_pairs.iloc[0])} 个来源独占 pair、{int(peryton.shared_pairs.iloc[0])} 个共享 pair。三 seed 平均 Popularity head MRR={peryton[peryton.side=='head'].popularity_mrr.mean():.4f}，tail MRR={peryton[peryton.side=='tail'].popularity_mrr.mean():.4f}；tail 目标落入候选度数最高十分位的比例为 {peryton[peryton.side=='tail'].test_fraction_in_top_decile.mean():.1%}。这与其 tail 排名优势一致。",
            f"共同 log2(head degree)×log2(tail degree) 区间只覆盖 Peryton 可评估独占 pair 的 {pp.coverage.min():.1%}–{pp.coverage.max():.1%}。该区间按三来源合并分布标准化后的 pair 平均 MRR 为 {pp.common_degree_standardized_mrr.mean():.4f}，应与同一支持区间的未加权结果 {pp.common_support_unweighted_mrr.mean():.4f} 一起看；不是对完整测试集的因果分解。",
            '匹配对照的新旧结果按配置分开合并，见 matched_mean_std.csv；记录对照与未知回退不合成一个标题指标。',
            '20 个候选案例没有完成独立证据核验，本轮采用删除发现性案例段落的方案，不将它们表述为生物学发现。',
            '严格未见实体归纳、公开数据许可与 Zenodo DOI 不由实验分数自动解决；见 release/。']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--bootstrap',type=int,default=2000); args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    base,pending=records(); df,summary=tables(base,pending); matched_tables()
    print('Head/tail tables complete',flush=True)
    paired=paired_analyses(base,args.bootstrap); print('Paired bootstrap complete',flush=True)
    source_audit(base); narrative(summary,paired,pending)
    write_json(REPORT/'analysis_receipt.json',dict(status='complete',n_boot=args.bootstrap,
        pending_results=len(pending),inputs={r['result_file']:sha256(ROOT/r['result_file']) for r in base.values()}))
    print('Wrote',REPORT,flush=True)
