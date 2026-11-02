"""Favorita: select minimum validation MSE before test evaluation."""
from pathlib import Path
import argparse
import json
from .study import execute


def configuration():
    return json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['GSE-VAR','Linear-SGL','cMLP','all'],default='all')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();config=configuration()
    for method in (['GSE-VAR','cMLP','Linear-SGL'] if args.method=='all' else [args.method]):
        if method=='cMLP':
            from .cmlp import main as run_cmlp
            run_cmlp(['--project',str(args.project),'--data',str(args.project/'data/favorita/cmlp_inputs.npz'),
                      '--device',args.device])
        else:
            print(execute(args.project,config,method,device=args.device if method=='GSE-VAR' else 'cpu'))


if __name__=='__main__': main()
