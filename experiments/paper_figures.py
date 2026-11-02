"""Render manuscript figures from preserved result archives without training."""
from pathlib import Path
import argparse
import csv
import io
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .shared import score_metrics
from .lorenz96.run import path_metrics

METHODS=('GSE-DisARM','NGC','GVAR','Linear-VAR')
LABEL={'GSE-DisARM':'GSE-VAR','NGC':'NGC','GVAR':'GVAR','Linear-VAR':'Linear VAR'}
COLORS=dict(zip(METHODS,['#087F8C','#B28C00','#D55E00','#777777']))


def rows(z,name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))


def arrays(z,name):
    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def save(fig,out,name):
    for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def favorita_path_diagram(data,output):
    """Write a vector diagram with signed medians from the saved coefficients."""
    positions={
        'DAIRY':('dairy','乳製品',-4,2.2),
        'BREAD/BAKERY':('bread','パン・ベーカリー',.3,3.2),
        'DELI':('deli','デリ食品',.3,1.9),
        'MEATS':('meat','肉類',.3,.6),
        'SEAFOOD':('seafood','魚介',5,.6),
        'FROZEN FOODS':('frozen','冷凍食品',-4,-1),
        'LIQUOR,WINE,BEER':('liquor','酒類',.3,-1),
        'BEVERAGES':('beverages','飲料',-4,-2.5),
        'PREPARED FOODS':('prepared','調理済食品',5,-2.5)}
    routes={('DAIRY','BREAD/BAKERY'):('east','west','above'),
            ('DAIRY','DELI'):('east','west','below'),
            ('DAIRY','MEATS'):('south east','north west','below,sloped'),
            ('MEATS','SEAFOOD'):('east','west','above'),
            ('FROZEN FOODS','LIQUOR,WINE,BEER'):('east','west','above'),
            ('LIQUOR,WINE,BEER','SEAFOOD'):('east','south west','below,sloped'),
            ('BEVERAGES','PREPARED FOODS'):('east','west','above')}
    edges=[]
    for lag,target,source in zip(*np.where(data['gate']!=0)):
        if target==source:continue
        key=str(data['names'][source]),str(data['names'][target])
        values=data['coefficients'][:,lag,target,source]
        edges.append(dict(source=key[0],target=key[1],lag_days=data['gate'].shape[0]-int(lag),
            coefficient_median=float(np.median(values)),
            absolute_coefficient_median=float(np.median(np.abs(values))),
            positive_fraction=float(np.mean(values>0)),negative_fraction=float(np.mean(values<0))))
    if {tuple(e[k] for k in ('source','target')) for e in edges}!=set(routes):
        raise ValueError('Saved routes differ from the diagram layout; update node positions.')
    lines=[r'\begin{tikzpicture}[box/.style={draw=violet!80!black,fill=violet!5,rounded corners=2pt,minimum width=3.1cm,minimum height=.65cm,font=\small},route/.style={-{Stealth[length=2.5mm]},line width=.9pt,draw=violet!80!black},every edge quotes/.style={font=\footnotesize}]']
    for node,label,x,y in positions.values():
        lines.append(r'\node[box] ('+node+') at ('+str(x)+','+str(y)+') {'+label+'};')
    for edge in edges:
        source,target=edge['source'],edge['target']
        start,end,where=routes[source,target]
        mixed=max(edge['positive_fraction'],edge['negative_fraction'])<.8
        style='route,dashed,draw=gray' if mixed else 'route'
        color='gray' if mixed else 'violet!80!black'
        label=r'\('+f"{edge['coefficient_median']:+.3f}"+r'\)'
        if edge['lag_days']!=1:label+='（'+str(edge['lag_days'])+'日前）'
        lines.append(r'\draw['+style+'] ('+positions[source][0]+'.'+start+') -- node['+where+',font=\\footnotesize,fill=white,inner sep=1pt,text='+color+'] {'+label+'} ('+positions[target][0]+'.'+end+');')
    lines.append(r'\end{tikzpicture}')
    (output/'favorita_paths.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (output/'favorita_paths.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(edges[0]));writer.writeheader();writer.writerows(edges)


def render(project,archive=None,favorita=None,output=None,cmlp=None):
    project=Path(project)
    out=Path(output) if output else project/'thesis/figures'
    out.mkdir(parents=True,exist_ok=True)
    archive=Path(archive) if archive else project/'results_lightweight.zip'
    favorita=Path(favorita) if favorita else project/'results/11_Favorita/11_Favorita.zip'
    cmlp=Path(cmlp) if cmlp else project/'results/11_Favorita_cMLP/full_756d4410363d9050.zip'
    records=[];paths=[];graphs={};truths={}
    with zipfile.ZipFile(archive) as z:
        for method in METHODS[:2]:
            points=rows(z,f'12_Lorenz96/additional/{method}/lambda_path_points.csv')
            for length in (250,500):
                for replicate in range(5):
                    part=[{k:float(r[k]) for k in ('path_value','recall','precision','fpr','edges')}
                          for r in points if r['phase']=='main' and int(r['T'])==length and int(r['replicate'])==replicate]
                    part.sort(key=lambda r:-r['path_value'])
                    metric=path_metrics(part)
                    records.append(dict(T=length,replicate=replicate,method=LABEL[method],**metric))
                    paths.append((length,method,[r['recall'] for r in part],
                                  [r['precision'] if r['edges'] else 1. for r in part]))
        for entry in json.loads(z.read('12_Lorenz96/selected_index.json')):
            if entry['method'] not in METHODS[:2]:continue
            a=arrays(z,'12_Lorenz96/'+entry['arrays'])
            key=int(entry['T']),int(entry['replicate']),entry['method']
            graphs[key]=a['graph'];truths[key[0]]=a['truth']
        for entry in json.loads(z.read('12_Lorenz96/baseline_index.json')):
            method=entry['method'];a=arrays(z,'12_Lorenz96/'+entry['arrays'])
            key=int(entry['T']),int(entry['replicate']),method
            graphs[key]=a['graph'];truths[key[0]]=a['truth']
            mask=~np.eye(len(a['truth']),dtype=bool);truth=a['truth'][mask]!=0
            score=(-a['pvalues'] if method=='Linear-VAR' else a['strength'])[mask]
            metric=score_metrics(truth,score)
            records.append(dict(T=key[0],replicate=key[1],method=LABEL[method],**metric))
            order=np.argsort(-score,kind='stable');y,s=truth[order],score[order]
            ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y)[ends]
            paths.append((key[0],method,np.r_[0,tp/truth.sum()],np.r_[1,tp/(ends+1)]))
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for ax,length in zip(axes,(250,500)):
            for method in METHODS:
                for n,(t,m,r,p) in enumerate(q for q in paths if q[0]==length and q[1]==method):
                    mean=np.mean([r['auprc'] for r in records if r['T']==length and r['method']==LABEL[method]])
                    ax.plot(r,p,color=COLORS[method],lw=1,alpha=.6,
                            label=f'{LABEL[method]}: {mean:.4f}' if n==0 else None)
            ax.set(title=f'T = {length}',xlabel='Recall',ylabel='Precision',xlim=(-.02,1.02),ylim=(-.02,1.04))
            ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.2)
        save(fig,out,'lorenz_lambda_paths')
        fig,axes=plt.subplots(2,5,figsize=(12,4.8),layout='constrained')
        for row,length in enumerate((250,500)):
            matrices=[truths[length]]+[np.mean([graphs[length,r,m] for r in range(5)],axis=0) for m in METHODS]
            for ax,a,label in zip(axes[row],matrices,['Reference']+[LABEL[m] for m in METHODS]):
                a=np.array(a,dtype=float);np.fill_diagonal(a,np.nan)
                ax.imshow(a,vmin=0,vmax=1,cmap='Blues');ax.set_title(label,fontsize=9)
                ax.set(xticks=[0,9,19],xticklabels=['1','10','20'],yticks=[0,9,19],yticklabels=['1','10','20'])
                ax.tick_params(labelsize=8)
            axes[row,0].set_ylabel(f'T = {length}\nTarget')
        save(fig,out,'lorenz_structures')
        index=json.loads(z.read('13_LinearVAR/main_index.json'))
        methods=('GSE-DisARM','GVAR','Linear-VAR');saved={};data={}
        for length in (250,500):
            for method in methods:
                e=next(v for v in index if v['T']==length and v['replicate']==0 and v['method']==method)
                saved[length,method]=arrays(z,'13_LinearVAR/'+e['result'])
                data[length]=arrays(z,'13_LinearVAR/'+e['dataset'])
        mask=~np.eye(4,dtype=bool)
        limit=max(np.max(np.abs(v['coefficients'][:,:,mask])) for v in saved.values())
        limit=max(limit,max(np.max(np.abs(v['true_standardized'][mask])) for v in data.values()))*1.08
        fig,axes=plt.subplots(2,3,figsize=(11,5.5),layout='constrained')
        for row,length in enumerate((250,500)):
            truth=data[length]['true_standardized'];pairs=list(zip(*np.where(mask&(truth!=0))))
            colors=['#3E1586','#D55E00','#087F8C','#2F73B8']
            for ax,method in zip(axes[row],methods):
                c=saved[length,method]['coefficients'][:,0]
                for i,j in zip(*np.where(mask&(truth==0))):ax.plot(c[:,i,j],color='#999999',lw=.65)
                for (i,j),color in zip(pairs,colors):
                    ax.plot(c[:,i,j],color=color,lw=.85);ax.axhline(truth[i,j],color=color,ls='--',lw=.7)
                ax.set(title=f'T = {length}: {LABEL[method]}',ylim=(-limit,limit),xlabel='Time index');ax.grid(alpha=.15)
            axes[row,0].set_ylabel('Effective coefficient')
        names=['x','w','y','z']
        handles=[Line2D([],[],color=color,label=f'{names[j]} -> {names[i]}') for (i,j),color in zip(pairs,colors)]
        handles += [Line2D([],[],color='#999999',label='Non-edges'),Line2D([],[],color='black',ls='--',label='True coefficient')]
        fig.legend(handles=handles,ncol=3,loc='outside upper center',fontsize=9,frameon=False)
        save(fig,out,'linear_coefficients')
    with zipfile.ZipFile(favorita) as z:
        aa=[arrays(z,m+'_evaluation.npz') for m in ('GSE-DisARM','Linear-SGL')]
        metrics=rows(z,'evaluation_metrics.csv')
        with zipfile.ZipFile(cmlp) as comparison:
            other=arrays(comparison,'cMLP_evaluation.npz')
            for key in ('dates','names','observed'):
                np.testing.assert_array_equal(other[key],aa[0][key])
            other['gate']=other['lag_support']
            aa.insert(1,other)
            metrics.insert(1,rows(comparison,'evaluation_metrics.csv')[0])
        labels=['GSE-VAR','cMLP','Linear VAR (SGL)']
        names=aa[0]['names'];mse=[np.mean((a['predictions']-a['observed'])**2,axis=0) for a in aa]
        for v,row in zip(mse,metrics):assert np.isclose(v.mean(),float(row['test_mse']),rtol=1e-6)
        fig,ax=plt.subplots(figsize=(9,5.6),layout='constrained');y=np.arange(12)
        for delta,v,label,color in zip([-.24,0,.24],mse,labels,['#087F8C','#B28C00','#777777']):
            ax.barh(y+delta,v,height=.22,label=label,color=color)
        ax.set(yticks=y,yticklabels=names,xlabel='Test MSE (standardized sales)');ax.invert_yaxis();ax.legend()
        save(fig,out,'favorita_target_mse')
        fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
        for ax,a,label in zip(axes,aa,labels):
            g=(a['gate']!=0).any(0).astype(float);np.fill_diagonal(g,np.nan)
            ax.imshow(g,vmin=0,vmax=1,cmap='Purples')
            ax.set(xticks=range(12),yticks=range(12),xticklabels=names,yticklabels=names,title=label,xlabel='Source',ylabel='Target')
            ax.tick_params(labelsize=7);plt.setp(ax.get_xticklabels(),rotation=90)
        save(fig,out,'favorita_structures')
        favorita_path_diagram(aa[0],out)
    (out/'source_checks.json').write_text(json.dumps(dict(
        archive=str(archive),favorita=str(favorita),cmlp=str(cmlp),lorenz_records=records,
        favorita_records=metrics,
        plotted_methods=[LABEL[m] for m in METHODS],
        interpretation='Preserved evaluations only; no fitting or result modification.'),indent=2),encoding='utf-8')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--archive',type=Path);p.add_argument('--favorita',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--cmlp',type=Path)
    a=p.parse_args();print(render(a.project,a.archive,a.favorita,a.output,a.cmlp))


if __name__=='__main__':main()
