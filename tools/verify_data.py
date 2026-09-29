"""Check an external benchmark directory against its own file manifest."""
from pathlib import Path
import argparse,hashlib,json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8 << 20),b''):h.update(block)
    return h.hexdigest()

def safe_path(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError(f'Path escapes data root: {relative}')
    return path

def verify(root):
    root=Path(root).resolve();manifest=root/'manifest.json'
    data=json.loads(manifest.read_text(encoding='utf-8'));files=data['files']
    if not files:raise ValueError('An empty file manifest cannot validate a dataset')
    failures=[]
    for name,value in files.items():
        expected=value['sha256'] if isinstance(value,dict) else value
        p=safe_path(root,name);actual=sha(p) if p.is_file() else None
        if actual!=expected:failures.append({'file':name,'expected':expected,'actual':actual})
    return {'passed':not failures,'files_checked':len(files),'manifest_sha256':sha(manifest),
            'graph_sha256':data.get('original_edges_sha256'),'failures':failures}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True)
    args=ap.parse_args();result=verify(args.data);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
