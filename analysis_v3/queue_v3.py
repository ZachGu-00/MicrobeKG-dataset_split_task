"""Generic job queue: python queue_v3.py jobs.json --workers N.  Each job: {"name", "cmd": [...], "done": glob, "cwd"}.
Jobs whose `done` glob already matches are skipped; a job with a `running` pattern waits while a process matching
it (pgrep -f) is alive, so a restarted queue never duplicates a job that an earlier queue left running.  Before starting a job the queue waits until GPU memory in use
is below --max-gpu-mib, so it can share the card with other queues."""
import argparse
import glob
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def gpu_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-gpu-mib", type=int, default=24000)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    jobs = [j for j in json.load(open(a.jobs)) if not glob.glob(j["done"])]
    print(len(jobs), "jobs", flush=True)

    def busy(j):
        pat = j.get("running")
        return bool(pat) and subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0

    def go(j):
        while busy(j):                      # an orphan of an earlier queue is still running this job
            time.sleep(60)
        if glob.glob(j["done"]):
            return f"SKIP {j['name']} (finished by an earlier queue)"
        while gpu_used() > a.max_gpu_mib:
            time.sleep(30)
        t0 = time.perf_counter()
        try:
            p = subprocess.run(j["cmd"], capture_output=True, text=True, timeout=a.timeout, cwd=j.get("cwd"))
            rc, tail = p.returncode, (p.stdout[-400:] + p.stderr[-1500:]) if p.returncode else ""
        except subprocess.TimeoutExpired:
            rc, tail = -9, "timeout"
        return f"DONE {j['name']} rc={rc} {time.perf_counter() - t0:.0f}s" + ("\n" + tail if tail else "")

    with ThreadPoolExecutor(a.workers) as ex:
        for i, msg in enumerate(ex.map(go, jobs)):
            print(msg, flush=True)
            time.sleep(0)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
