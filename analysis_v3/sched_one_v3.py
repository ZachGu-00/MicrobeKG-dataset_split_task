"""Run one PyKEEN variant cell with the configuration tuned for its parent setting: sched_one_v3.py SETTING MODEL SEED."""
import subprocess
import sys

sys.path.insert(0, "/root/autodl-tmp/mkg")
from sched_variants_v3 import cmd_of  # noqa: E402

s, m, sd = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.exit(subprocess.run(cmd_of(s, m, sd)).returncode)
