"""Check the staged/tracked publication tree contains code and configuration only."""
from pathlib import Path
import ast,json,re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.toml','.sha256','.py','.cpp','.cu','.h','.cuh','.md','.json','.txt','.yml','.yaml','.sh','.diff'}
SPECIAL={'.gitignore','.gitattributes','LICENSE','NOTICE'}
FORBIDDEN_DIRS={'original_data','preprocessed','data','data_local','benchmark_data','supplement_data','revision_data',
               'kg_build','kg_build_audited_20260928','kg_build_corrected_20260928','paper_work_collaboration',
               'results','reference_results','revision_results','archive','release_staging','literature_evidence','autodl','outputs'}
SECRET_PATTERNS=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']

def main():
    result=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
    files=[p for p in result.stdout.decode('utf-8').split('\0') if p];errors=[];total=0;python_files=0
    for name in files:
        p=ROOT/name;parts=Path(name).parts
        if not p.is_file():errors.append(f'Missing indexed file: {name}');continue
        if p.is_symlink():errors.append(f'Symlink in publication tree: {name}')
        if parts[0] in FORBIDDEN_DIRS:errors.append(f'Data/private directory: {name}')
        if p.suffix.lower() not in ALLOWED and p.name not in SPECIAL:errors.append(f'Unexpected file type: {name}')
        total+=p.stat().st_size
        if p.stat().st_size>2_000_000:errors.append(f'Oversized source file: {name}')
        text=p.read_text(encoding='utf-8-sig')
        if '\0' in text:errors.append(f'Binary payload: {name}')
        if p.suffix=='.py':
            try:ast.parse(text,filename=name);python_files+=1
            except SyntaxError as e:errors.append(f'Syntax error: {name}:{e.lineno}')
        for pattern in SECRET_PATTERNS:
            if re.search(pattern,text):errors.append(f'Credential/private-key pattern: {name}')
        if re.search(r'[CD]:[\\/](?:Users|Paperwork_code)[\\/]',text):errors.append(f'Workstation path: {name}')
    summary={'files':len(files),'bytes':total,'python_files':python_files,'errors':errors}
    print(json.dumps(summary,indent=2));return bool(errors)

if __name__=='__main__':sys.exit(main())
