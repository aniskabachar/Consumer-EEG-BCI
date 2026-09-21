"""
fix_structure.py -- run once from BCI/ root:
    python fix_structure.py
Moves flat .py files into proper folders, adds __init__.py.
"""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
print("Root:", root)

mapping = {
    'models':          ['bci_model.py'],
    'data_processing':  ['preprocessor.py', 'debug_mat.py'],
    'training':         ['trainer.py'],
    'evaluation':       ['evaluate.py'],
    'deployment':       ['realtime_inference.py'],
}

for folder, files in mapping.items():
    fdir = root / folder
    fdir.mkdir(exist_ok=True)
    init = fdir / '__init__.py'
    if not init.exists():
        init.write_text('')
    for fname in files:
        src = root / fname
        dst = fdir / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print("moved", fname, "->", folder + "/")
        elif dst.exists():
            print("=", folder + "/" + fname, "already in place")
        else:
            print("!", fname, "not found at root, skip")

(root / '__init__.py').write_text('')

for d in ['saved_models', 'processed_data']:
    (root / d).mkdir(exist_ok=True)

print("\nDone. main.py stays at root.")
print("Run:  python main.py preprocess --data_dir ./SEED-VIG --output_dir ./processed_data")
