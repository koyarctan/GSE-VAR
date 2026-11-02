"""Linear VAR: signed-structure recovery with independent pilot selection."""
from pathlib import Path
import argparse
from ..synthetic import SyntheticExperiment,default_config


def configuration(mode='full',device='cpu'):
    return default_config('linear_var',mode,device)


def run(project,config):
    experiment=SyntheticExperiment(project,'linear_var',config)
    with experiment.run.lock():
        choices=experiment.tune()
        experiment.evaluate_main(choices)
    return experiment.run.path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['full','smoke'],default='full')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    print(run(args.project,configuration(args.mode,args.device)))


if __name__=='__main__': main()
