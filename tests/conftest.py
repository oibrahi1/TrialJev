import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Processed datasets are not in the repository (see DATA.md). Point this at a directory that
# holds tm-v1-b4096/{train,development,test}.jsonl and splits.json to run the data tests.
DATA_DIR = Path(os.environ.get("TRIALMATCH_DATA", ROOT / "data"))
