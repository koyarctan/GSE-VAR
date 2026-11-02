"""Summarize completed experiment CSVs without fitting models."""
from pathlib import Path
import argparse
import pandas as pd


def summarize(path):
    path=Path(path)
    source=path/'main_metrics.csv'
    if not source.exists():source=path/'evaluation_metrics.csv'
    frame=pd.read_csv(source)
    if 'status' in frame and (frame.status!='ok').any():
        raise ValueError('Resolve incomplete evaluations before aggregation.')
    fields=[k for k in ['auprc','auroc','f1','ba_pos','ba_neg','mean_sign_ba','test_mse','test_mae','nonself_edges'] if k in frame]
    keys=['T','method'] if 'T' in frame else ['method']
    result=frame.groupby(keys,sort=False)[fields].agg(['count','mean','std'])
    result.to_csv(path/'summary.csv')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory',type=Path)
    print(summarize(parser.parse_args().result_directory).to_string())


if __name__=='__main__':main()
