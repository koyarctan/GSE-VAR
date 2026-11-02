"""Lorenz96: independent pilot selection and regularization-path evaluation."""
from pathlib import Path
import argparse
import numpy as np
from ..synthetic import SyntheticExperiment, default_config
from ..shared import write_csv, write_json


def configuration(mode='full', device='cpu'):
    config = default_config('lorenz96', mode, device)
    config['regularization_grid'] = {
        '250': {'GSE-VAR': [0,.3,1,2,3,4,5,7,10,30],
                'NGC': [0,.1,.25,.35,.45,.5,.6,.75,1,10]},
        '500': {'GSE-VAR': [0,.3,1,2,3,4,5,7,10,30],
                'NGC': [0,.05,.1,.15,.25,.35,.4,.5,1,10]}}
    return config


def path_metrics(rows):
    rows = sorted(rows, key=lambda r: -r['path_value'])
    recall = np.array([r['recall'] for r in rows])
    precision = np.array([r['precision'] if r['edges'] else 1. for r in rows])
    fpr = np.array([r['fpr'] for r in rows])
    if np.any(np.diff(recall) < 0) or np.any(np.diff(fpr) < 0):
        raise ValueError('Nonmonotone regularization path; inspect points before integration.')
    return dict(auprc=float(np.sum(np.diff(recall)*(precision[:-1]+precision[1:])/2)),
                auroc=float(np.sum(np.diff(fpr)*(recall[:-1]+recall[1:])/2)),
                recall_min=float(recall[0]), recall_max=float(recall[-1]),
                fpr_min=float(fpr[0]), fpr_max=float(fpr[-1]))


def run(project, config):
    experiment = SyntheticExperiment(project, 'lorenz96', config)
    with experiment.run.lock():
        choices = experiment.tune()
        records = experiment.evaluate_main(choices)
        points=[]
        for length in config['lengths']:
            for replicate, seed in enumerate(config['main_seeds']):
                case=experiment.case(length, seed)
                for method in ('GSE-VAR','NGC'):
                    selected=choices[str(length)][method]
                    candidates=[p for p in experiment.candidates(method,length) if p['hidden']==selected['hidden']]
                    part=[]
                    for profile in candidates:
                        metrics,_=experiment.evaluate(case,method,profile,42+replicate)
                        value=profile['budget'] if method=='GSE-VAR' else profile['lam']
                        row=dict(T=length,replicate=replicate,method=method,path_value=value,**metrics)
                        part.append(row);points.append(row)
                    record=next(r for r in records if r['T']==length and r['replicate']==replicate and r['method']==method)
                    record.update(path_metrics(part))
        write_csv(experiment.run.path/'lambda_path_points.csv',points)
        write_csv(experiment.run.path/'main_metrics.csv',records)
        write_json(experiment.run.path/'evaluation_protocol.json',dict(
            self_edges=False,selection='independent pilot mean F1',
            path='fixed selected width; regularization sweep for GSE-VAR and NGC',
            integration='trapezoidal over observed points; missing endpoints are not extrapolated',
            baselines='coefficient-strength or negative F-test p-value thresholds'))
    return experiment.run.path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['full','smoke'],default='full')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    print(run(args.project,configuration(args.mode,args.device)))


if __name__=='__main__': main()
