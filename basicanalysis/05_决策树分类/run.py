from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import classification
from sklearn.tree import DecisionTreeClassifier
classification(Path(__file__).resolve().parent,
 [(f'depth={depth}',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=100,random_state=42)) for depth in [3,5,8]],'决策树')
