from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import classification
from sklearn.linear_model import LogisticRegression
classification(Path(__file__).resolve().parent,
 [(f'C={c}',LogisticRegression(C=c,max_iter=3000,random_state=42)) for c in [.1,1.,10.]],'逻辑回归',standardize=True)
