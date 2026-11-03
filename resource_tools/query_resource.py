"""Read-only MicrobeKG evidence queries using retained local inputs (no network/GPU)."""
from pathlib import Path
import argparse,csv,hashlib,json,re,sys,zipfile,struct,zlib,xml.etree.ElementTree as ET
csv.field_size_limit(10**9)
SOURCES={'BugSigDB':'bugsigdb_edges.tsv','Disbiome':'disbiome_edges.tsv','GMMAD':'gmmad_edges.tsv','GMrepo_v3':'gmrepo_disease_edges.tsv','HMDAD':'hmdad_edges.tsv','Lit44':'literature44_edges.tsv','Peryton':'peryton_edges.tsv','gutMDisorder_v2':'gutmdisorder_disease_edges.tsv','mBodyMap':'mbodymap_edges.tsv','cross_source_conflict':'cross_source_conflict_edges.tsv'}
GRAPH_DIRECTORY='kg_build_audited'
AUDITED_SUPPORT=None

def graph_file(root,relative_path):
    return local(root,GRAPH_DIRECTORY)/relative_path

def source_file(root,name):
    candidate=graph_file(root,'edges/'+name)
    return candidate if candidate.exists() else root/'kg_build/edges'/name
def rows(p,delimiter='\t',skip_comments=False):
    with p.open(encoding='utf-8-sig',newline='') as f:
        lines=(x for x in f if not x.startswith('#')) if skip_comments else f
        yield from csv.DictReader(lines,delimiter=delimiter)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def relative(root,p):return p.relative_to(root).as_posix()
def local(root,rel):
    p=(root/rel).resolve()
    if not p.is_relative_to(root):raise ValueError('Input path escapes project')
    return p
def association(root,microbe,disease):
    found=[]
    for i,r in enumerate(rows(graph_file(root,'final_kg_edges.tsv')),1):
        if (r['head_type'],r['head_id'],r['tail_type'],r['tail_id'])==('microbe',microbe,'disease',disease):found.append(dict(graph_data_row=i,**r))
    source_rows=[]
    active_sources={r['relation']:set(r['source'].split('|')) for r in found}
    audited=graph_file(root,'provenance/literature_release_ledger.tsv').exists()
    support_root=local(root,AUDITED_SUPPORT) if AUDITED_SUPPORT else graph_file(root,'provenance/query_support')
    support_rows=[];curated_rows=[]
    if audited:
        support_path=support_root/'lit/literature_edge_support.tsv'
        curated_path=support_root/'curated/contrast_labels_all.tsv'
        if not support_path.exists() or not curated_path.exists():
            raise ValueError('Audited queries require lit/literature_edge_support.tsv and curated/contrast_labels_all.tsv under '+str(support_root)+'; select a retained location with --support-dir')
        support_rows=[r for r in rows(support_path) if r['head_id']==microbe and r['tail_id']==disease and 'Lit44' in active_sources.get(r['relation'],set())]
        curated_rows=[r for r in rows(curated_path) if r['taxon']==microbe and r['disease']==disease and r['source'] in active_sources.get(r['direction_for_D'],set())]
    fall={r['original_id']:r['ncbi_taxon_id'] for r in rows(graph_file(root,'mapping/microbe_name_fallback.tsv')) if r['ncbi_taxon_id']}
    for s in sorted({s for r in found for s in r['source'].split('|')}):
        if s not in SOURCES:continue
        p=source_file(root,SOURCES[s])
        for i,r in enumerate(rows(p),1):
            if fall.get(r['head_id'],r['head_id'])==microbe and r['tail_id']==disease and s in active_sources.get(r['relation'],set()):
                source_rows.append(dict(source=s,standardized_file=relative(root,p),standardized_data_row=i,raw_record=r,
                  upstream_locator_status='not retained in this standardized record; no original-table coordinate inferred'))
        if s=='BugSigDB':
            p=source_file(root,'bugsigdb_conflict_edges.tsv')
            for i,r in enumerate(rows(p),1):
                if r['head_id']==microbe and r['tail_id']==disease:source_rows.append(dict(source=s,standardized_file=relative(root,p),standardized_data_row=i,raw_record=r,upstream_locator_status='derived conflict; original signature IDs retained in evidence'))
    locators=[]
    peryton_keys={re.search(r'S.No (\d+)',r['record_key']).group(1) for r in curated_rows if r['source']=='Peryton' and re.search(r'S.No (\d+)',r['record_key'])}
    # Peryton: join explicit taxonomy, MeSH and relationship fields of the retained download.
    p=root/'original_data/Peryton-results.tsv'
    if p.exists():
        for i,r in enumerate(rows(p),1):
            if (r['S.No'] in peryton_keys if audited else 'NCBI_TAXON:'+r['microbe_ncbi_tax_id']==microbe and r['disease_mesh_id'].removeprefix('MESH:')==disease):
                locators.append(dict(source='Peryton',file=relative(root,p),data_row=i,source_record_id=r['S.No'],direction=r['relationship_name'],pmid=r['publication_PMID'],microbe=r['microbe_scientific_name'],disease=r['disease_name'],body_site=r['samples_origin_name'],check='exact retained taxon and MeSH fields'))
    # BugSigDB: signature IDs preserve a source-owned lookup key, including non-PMID study IDs.
    wanted=set()
    for r in source_rows:
        if r['source']=='BugSigDB':wanted.update(re.findall(r'bsdb=(bsdb:[^|;]+)',r['raw_record'].get('evidence','')))
    if audited:wanted={r['record_key'] for r in curated_rows if r['source']=='BugSigDB'}
    p=root/'original_data/BugSigDBE.csv'
    if p.exists() and wanted:
        for i,r in enumerate(rows(p,',',True),1):
            if any(k in r.get('BSDB ID','') or k.removeprefix('bsdb:')==r.get('BSDB ID','') for k in wanted):
                locators.append(dict(source='BugSigDB',file=relative(root,p),data_row=i,source_record_id=r.get('BSDB ID'),direction=r.get('Abundance in Group 1'),pmid=r.get('PMID'),taxa=r.get('NCBI Taxonomy IDs'),condition=r.get('Condition'),group1=r.get('Group 1 name'),group0=r.get('Group 0 name'),check='signature ID join; group labels and disease mapping remain visible'))
    if audited:
        wanted_disbiome={r['record_key'].removeprefix('experiment_id='):r for r in curated_rows if r['source']=='Disbiome'}
        p=root/'original_data/disbiome_experiment_data.txt'
        if wanted_disbiome and p.exists():
            for i,r in enumerate(json.loads(p.read_text(encoding='utf-8')),1):
                key=str(r['experiment_id'])
                if key in wanted_disbiome:
                    locators.append(dict(source='Disbiome',file=relative(root,p),json_array_index_zero_based=i-1,source_record_id=key,
                        original_record=r,audited_assignment=wanted_disbiome[key],check='exact source experiment_id joined to retained disease and direction assignment'))
    # Literature: recover parsed-row coordinates and verify exact taxon text in XLSX.
    resolution={r['raw']:r for r in rows(root/'literature_evidence/kg_ready/literature44_microbe_resolution.tsv')}
    res={k:r['ncbi_taxon_id'] for k,r in resolution.items()}
    lraw=[x['raw_record'] for x in source_rows if x['source']=='Lit44'];pmids={p for x in lraw for p in re.findall(r'pmid=(\d+)',x['evidence'])}
    supp={}
    for p in (root/'literature_evidence/literature').rglob('*'):
        if p.is_file():supp.setdefault(p.name,[]).append(p)
    workbooks={};ledger_path=graph_file(root,'provenance/literature_release_ledger.tsv' if audited else 'provenance/literature_record_ledger.tsv')
    corrections={(r['rec_id'],r['tail_id']):r for r in rows(ledger_path)} if ledger_path.exists() else {}
    if GRAPH_DIRECTORY!='kg_build' and not ledger_path.exists():raise ValueError('Corrected literature query requires its record ledger')
    original_rows=list(rows(root/'literature_evidence/extracted/edges_for_kg.tsv'))
    if audited:
        literature_records=[]
        for support in support_rows:
            rid=support['rec_id'];match=re.fullmatch(r'R(\d+)',rid)
            original=original_rows[int(match.group(1))] if match and int(match.group(1))<len(original_rows) else {}
            literature_records.append((rid,dict(original,**support),support,original))
    else:literature_records=[('R%05d'%i,x,None,x) for i,x in enumerate(original_rows)]
    for record_id,x,support,original in literature_records:
        correction=corrections.get((record_id,disease))
        if not audited and correction and correction['action']=='drop':continue
        original_stat={k:original.get(k,'') for k in ['stat_type','stat_value']}
        resolved_head=support['head_id'] if support else res.get(x['microbe_name_raw'])
        if correction and not audited:
            resolved_head=correction.get('new_head_id') or resolved_head
            x=dict(x)
            for field in ['stat_type','stat_value']:
                if correction.get('new_'+field):x[field]=correction['new_'+field]
        if x['pmid'] not in pmids or resolved_head!=microbe:continue
        relation=support['relation'] if support else {'enriched_in_case':'enriched_in','depleted_in_case':'depleted_in'}.get(x['direction'])
        wanted_fields={k:x[k] for k in ['pmid','stat_type','stat_value','source_sheet']}
        matched=False
        for y in lraw:
            if y['relation']!=relation:continue
            for entry in y['evidence'].split(';;'):
                fields=dict(part.split('=',1) for part in entry.split('|') if '=' in part)
                if all(fields.get(k,'')==v for k,v in wanted_fields.items()):matched=True
        if not matched:continue
        paths=[p for p in supp.get(x['source_file'],[]) if 'PMID_'+x['pmid'] in str(p)]
        rec=dict(source='Lit44',pmid=x['pmid'],file=relative(root,paths[0]) if len(paths)==1 else None,
          file_candidates=[relative(root,p) for p in paths],sheet=x['source_sheet'],parser_row_index_zero_based=int(x['row_index']),worksheet_row_one_based=int(x['row_index'])+1,
          raw_microbe=x['microbe_name_raw'],taxon_mapping=support if support else resolution[x['microbe_name_raw']],direction=x['direction'],direction_rule=x['direction_source'],stat_type=x['stat_type'],stat_value=x['stat_value'],comparison=x.get('comparison',''),
          disease_assignment='publication disease-set expansion; requires comparison-level review' if x['pmid'] in {'39604394','40520642','41081945','41361834'} else 'retained publication-level disease assignment',
          coordinate_status='parsed coordinate retained; original-cell check unavailable for this format')
        if correction:
            rec.update(corrected_record=correction,original_parsed_statistic=original_stat)
            if correction.get('new_stat_type') and not audited:
                rec['original_parsed_comparison']=rec['comparison'];rec['comparison']=correction['detail']
        if GRAPH_DIRECTORY!='kg_build':rec['disease_assignment']='corrected record-by-disease rules applied; correction ledger linked when changed'
        if audited:
            rec.update(disease_assignment='retained audited assertion support',support_file=relative(root,support_root/'lit/literature_edge_support.tsv'),
                       support_record_id=record_id,support_record=support,original_parsed_record=original)
        if len(paths)==1 and paths[0].suffix.lower()=='.xlsx':
            try:
                import openpyxl
                key=str(paths[0])
                if key not in workbooks:workbooks[key]=openpyxl.load_workbook(paths[0],read_only=True,data_only=True)
                ws=workbooks[key][x['source_sheet']];rn=int(x['row_index'])+1
                cells=list(next(ws.iter_rows(min_row=rn,max_row=rn,values_only=True)))
                exact=any(str(v).strip()==x['microbe_name_raw'].strip() for v in cells if v is not None)
                rec.update(original_row_values=cells,raw_taxon_cell_match=exact,coordinate_status='original XLSX taxon cell matched' if exact else 'coordinate mismatch; inspect source table')
            except (ImportError,KeyError,ValueError,StopIteration) as e:rec['coordinate_status']='unavailable: '+str(e)
        locators.append(rec)
    for w in workbooks.values():w.close()
    by_source={s:sum(x['source']==s for x in locators) for s in sorted({x['source'] for x in source_rows})}
    upstream=[]
    if audited:
        upstream=[dict(source=r['source'],record=r) for r in curated_rows]
        upstream += [dict(source='Lit44',record=r) for r in support_rows]
        upstream += [dict(source=r['source'],record=r['raw_record']) for r in source_rows if r['source'] not in {'Lit44','cross_source_conflict'} and r['source'] not in {x['source'] for x in curated_rows}]
    return dict(query=dict(microbe=microbe,disease=disease),graph_assertions=found,source_records=source_rows,upstream_records=upstream,
      derived_conflict_records=[r for r in source_rows if r['raw_record']['relation']=='inconsistent_association'],upstream_locations=locators,
      upstream_locations_per_source=by_source,interpretation='Directions are reported source associations. Locator availability differs by source; a missing locator is not fabricated. Confidence is source-assigned metadata, not a calibrated probability.')

def utilization(root,microbe,substrate):
    graph=[]
    for i,r in enumerate(rows(graph_file(root,'final_kg_edges.tsv')),1):
        if r['head_type']=='microbe' and r['head_id']==microbe and r['tail_type']=='substrate' and (not substrate or r['tail_id']==substrate) and r['relation'] in {'can_utilize','utilizes','does_not_utilize'}:
            graph.append(dict(graph_data_row=i,**r))
    experimental={r['tail_id'] for r in graph if r['relation']!='can_utilize'}
    if not substrate:graph=[r for r in graph if r['tail_id'] in experimental]
    ids={r['tail_id'] for r in graph};members={k:[] for k in ids};reverse={}
    for r in rows(graph_file(root,'mapping/metabolite_canonical.tsv')):
        if r['canonical_id'] in ids:members[r['canonical_id']].append(r['original_id']);reverse[r['original_id']]=r['canonical_id']
    njcdir=root/'original_data/Large-scale metabolic interaction network of the mouse and human gut microbiota/NJC19_network_data_and_code'
    network=json.loads((njcdir/'NJC19_network.json').read_text(encoding='utf-8'));org=json.loads((njcdir/'NJC19_organism.json').read_text(encoding='utf-8'));compound=json.loads((njcdir/'NJC19_compound.json').read_text(encoding='utf-8'))
    refs=json.loads((njcdir/'NJC19_reference.json').read_text(encoding='utf-8'));records=[]
    for r in graph:
        if r['relation']=='can_utilize':continue
        for key in re.findall(r'njc_id=([^|;]+)',r['evidence']):
            original=network[key];on=org[original['Species']];co=compound[original['Small-molecule metabolite or macromolecule']]
            activity=original['Metabolic activity']
            expected='does_not_utilize' if 'Consumption (import) (-)' in activity else ('utilizes' if 'Consumption (import)' in activity or activity=='Macromolecule degradation' else None)
            scope='degradation grouped under utilizes; broader than direct uptake' if activity=='Macromolecule degradation' else ('multiple activities; consumption evidence selected' if ', ' in activity else 'direct source activity')
            ref_ids=original['Ref. #']
            if isinstance(ref_ids,dict):ref_ids=ref_ids.get('Consumption (import) (-)' if expected=='does_not_utilize' else 'Consumption (import)',[])
            raw_ids=['KEGG:'+k for k in co.get('KEGG compound identifier',[])];mapped={reverse.get(k,k) for k in raw_ids}
            tax='NCBI_TAXON:'+str(on['NCBI Taxonomy ID'])
            records.append(dict(source='NJC19',original_file=relative(root,njcdir/'NJC19_network.json'),json_key=key,original_record=original,
              organism_file=relative(root,njcdir/'NJC19_organism.json'),original_taxon=tax,original_chemical_ids=raw_ids,canonical_id=r['tail_id'],
              relation_check=expected==r['relation'],relation_mapping_scope=scope,taxon_check=tax==microbe,chemical_mapping_check=r['tail_id'] in mapped,
              references={k:dict(reference=refs.get('Ref. #'+k.removesuffix('(G)')),evidence_level='genus-level information' if k.endswith('(G)') else 'no genus qualifier recorded') for k in ref_ids}))
    sbml=[];archive=root/'original_data/AGORA2_SBML.zip'
    # Show original SBML bounds for at most three model/reaction records; retain all graph evidence above.
    requests=[]
    for r in graph:
        if r['relation']=='can_utilize':
            requests.extend((r['tail_id'],*x) for x in re.findall(r'([^|;]+)\|(R_EX_[^|;]+)\|lb=([^|;]+)\|ub=([^|;]+)',r['evidence']))
    negative_ids={r['tail_id'] for r in graph if r['relation']=='does_not_utilize'}
    requests.sort(key=lambda x:(x[0] not in negative_ids,x[0],x[1],x[2]))
    with zipfile.ZipFile(archive) as z:
        for cid,model,reaction,lb,ub in requests[:3]:
            member=model+'.xml'
            try:data=z.read(member)
            except NotImplementedError:
                from inflate64 import Inflater
                info=z.getinfo(member)
                if info.compress_type!=9 or info.flag_bits&1:raise ValueError('Unsupported compressed or encrypted ZIP member')
                with archive.open('rb') as raw:
                    raw.seek(info.header_offset);header=raw.read(30)
                    if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
                    name_len,extra_len=struct.unpack_from('<HH',header,26);raw.seek(name_len+extra_len,1)
                    compressed=raw.read(info.compress_size)
                data=Inflater().inflate(compressed)
                if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('Decompressed SBML size/CRC mismatch')
            try:tree=ET.fromstring(data)
            except ET.ParseError:
                from lxml import etree
                tree=etree.fromstring(data,parser=etree.XMLParser(recover=True))
            params={e.attrib.get('id'):e.attrib.get('value') for e in tree.iter() if e.tag.split('}')[-1]=='parameter'}
            target=next(e for e in tree.iter() if e.tag.split('}')[-1]=='reaction' and e.attrib.get('id')==reaction)
            attrs={k.split('}')[-1]:v for k,v in target.attrib.items()};lo=params.get(attrs.get('lowerFluxBound'));hi=params.get(attrs.get('upperFluxBound'))
            sbml.append(dict(source='AGORA2',archive=relative(root,archive),member=member,member_sha256=hashlib.sha256(data).hexdigest(),reaction_id=reaction,canonical_substrate=cid,
              lower_bound=lo,upper_bound=hi,bounds_match_record=lo is not None and hi is not None and float(lo)==float(lb) and float(hi)==float(ub)))
    return dict(query=dict(microbe=microbe,substrate=substrate),graph_assertions=graph,canonical_members=members,njc19_original_records=records,sbml_original_records=sbml,
      sbml_records_shown=len(sbml),sbml_evidence_records_total=len(requests),
      interpretation='can_utilize records exchange-bound capacity; utilizes and does_not_utilize retain positive and negative experimental records. Same typed ID enables comparison, not equivalent study conditions or proof of biological contradiction.')

def main():
    global GRAPH_DIRECTORY,AUDITED_SUPPORT
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--graph-dir',default='kg_build_audited',help='Versioned graph directory relative to project root')
    ap.add_argument('--support-dir',help='Audited support directory relative to project root (default: GRAPH/provenance/query_support)')
    ap.add_argument('--mode',choices=['association','utilization','demo'],default='demo');ap.add_argument('--microbe');ap.add_argument('--disease');ap.add_argument('--substrate');args=ap.parse_args()
    root=args.project_root.resolve();out=args.out.resolve()
    GRAPH_DIRECTORY=args.graph_dir
    AUDITED_SUPPORT=args.support_dir
    if out.is_relative_to(local(root,GRAPH_DIRECTORY)):raise ValueError('Query output must not modify the graph directory')
    for protected in [p.name for p in root.glob('kg_build*') if p.is_dir()] + ['original_data','literature_evidence','results','preprocessed']:
        if out.is_relative_to(root/protected):raise ValueError('Write query output outside source and frozen-result directories')
    if AUDITED_SUPPORT and out.is_relative_to(local(root,AUDITED_SUPPORT)):raise ValueError('Query output must not modify support inputs')
    out.mkdir(parents=True,exist_ok=True);graph=graph_file(root,'final_kg_edges.tsv');before=sha(graph)
    result={'graph_sha256':before,'graph_directory':GRAPH_DIRECTORY,'mode':args.mode}
    flagpath=Path(__file__).resolve().with_name('quality_flags.json') if GRAPH_DIRECTORY=='kg_build' else graph_file(root,'quality_flags.json')
    flags=json.loads(flagpath.read_text(encoding='utf-8')) if flagpath.exists() else {}
    if flags and flags['graph_sha256']!=before:raise ValueError('Quality flags refer to a different graph version')
    if args.mode in ['association','demo']:
        if args.mode=='association' and not (args.microbe and args.disease):ap.error('association requires --microbe and --disease')
        result['association']=association(root,args.microbe or 'NCBI_TAXON:1017280',args.disease or 'D015179')
    if args.mode in ['utilization','demo']:
        if args.mode=='utilization' and not args.microbe:ap.error('utilization requires --microbe')
        result['utilization']=utilization(root,args.microbe or 'NCBI_TAXON:166486',args.substrate)
    if 'utilization' in result:
        ids=result['utilization']['canonical_members'];result['utilization']['mapping_review_flags']={k:flags.get('chemical_clusters',{}).get(k) for k in ids if k in flags.get('chemical_clusters',{})}
    result['quality_review_scope']=flags.get('scope','No quality flag manifest supplied')
    result['inputs_unchanged']=before==sha(graph);assert result['inputs_unchanged']
    (out/'queries.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    summary=dict(graph_sha256=before,inputs_unchanged=True)
    if 'association' in result:summary['association']={k:len(result['association'][k]) for k in ['graph_assertions','source_records','upstream_records','derived_conflict_records','upstream_locations']}
    if 'utilization' in result:
        u=result['utilization'];summary['utilization']=dict(graph_assertions=len(u['graph_assertions']),njc19_records=len(u['njc19_original_records']),sbml_records=len(u['sbml_original_records']),
          all_njc_checks=all(x['relation_check'] and x['taxon_check'] and x['chemical_mapping_check'] for x in u['njc19_original_records']),all_sbml_bounds_match=all(x['bounds_match_record'] for x in u['sbml_original_records']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
