from pathlib import Path
import importlib.util
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')
print({x: bool(importlib.util.find_spec(x)) for x in ['numpy','pandas','pyarrow','scipy','matplotlib','xlrd','sklearn','torch']})
print('disk', shutil.disk_usage(Path.cwd()))
print('python', sys.executable)
