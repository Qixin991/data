from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.deps'))
import pandas as pd
from baselines import yearly
out=ROOT/'results/development'
d=pd.read_csv(out/'validation_predictions.csv',dtype={'k4':str})
rows=[]
for name in d.columns[4:]:
    rows.extend([{'model':name,**r} for r in yearly(d,d[name].to_numpy())])
pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
print('Refreshed metrics from saved predictions only; no models refit.')
