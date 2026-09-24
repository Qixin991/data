"""python run_all.py；--skip-preparation 复用已完成的公共数据。"""
from pathlib import Path
import argparse,subprocess,sys
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--skip-preparation',action='store_true');args=p.parse_args()
scripts=[] if args.skip_preparation else [root/'00_数据预处理与描述统计'/'prepare.py']
scripts.append(root/'00_数据预处理与描述统计'/'describe.py')
scripts += sorted(root.glob('[0-9][1-9]_*/run.py'))
for script in scripts:
 print(f'运行 {script.relative_to(root)}',flush=True)
 subprocess.run([sys.executable,str(script)],check=True,cwd=root)
subprocess.run([sys.executable,str(root/'summarize.py')],check=True,cwd=root)
