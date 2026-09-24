from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import classification
from sklearn.ensemble import RandomForestClassifier
classification(Path(__file__).resolve().parent,
 [(f'depth={depth}',RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=30,max_features='sqrt',n_jobs=-1,random_state=42)) for depth in [5,10,None]],'随机森林')
