#!/usr/bin/env python3
"""
Summarise the scaling/smoke-test jobs created by 00_scaling_test.sh (run on the login node):
    python3 scaling_report.py [base_dir]          (default: $HPC_SCRATCH/<this folder name>)
For every scaling_* folder: cores, wall time, number of accepted steps, seconds per step, pseudo-time reached,
x_ct reached, and an extrapolated wall time for a full run using the step counts of the v2.0 runs.
Also usable for finished production runs (prints per-case progress).
"""
import glob
import os
import re
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("HPC_SCRATCH", "."), os.path.basename(os.path.dirname(os.path.abspath(__file__))))

# number of accepted time steps of the full v2.0 runs (2D vM, 32 cores) for extrapolation
STEPS_V2 = {0.25: 2870, 0.375: 6926, 0.4: 10072, 0.6: 7429, 0.75: 17669, 1.0: 16656, 1.4: 31679, 2.0: 69962, 3.0: 87464, 4.0: 65092}
if "3D" in os.path.basename(os.path.dirname(os.path.abspath(__file__))):
    STEPS_V2 = {1.0: 4116}  # only w_s = 1.0 of the v2.0 3D runs is documented locally (64 cores); others unknown


def wall_time(folder):
    """wall-clock seconds: from the newton log ('# elapsed time: [wall, user, sys]') or from the SLURM out timestamps"""
    log = os.path.join(folder, "run_simulation_log.txt")
    if os.path.exists(log):
        for line in reversed(open(log).read().splitlines()):
            m = re.search(r"elapsed time:\s*\[([0-9.eE+-]+)", line)
            if m:
                return float(m.group(1)), "log"
    outs = sorted(glob.glob(os.path.join(folder, "*.out.*")))
    if outs:
        txt = open(outs[-1]).read()
        import datetime
        stamps = re.findall(r"^\[(.+?)\] (job|fracture simulation|finished)", txt, flags=re.M)
        try:
            times = [datetime.datetime.strptime(s[0], "%a %b %d %H:%M:%S %Z %Y") for s in stamps]
            if len(times) >= 2:
                return (times[-1] - times[0]).total_seconds(), "slurm-out (whole job incl. meshing)"
        except Exception:
            pass
    return None, "running / no runtime found"


rows = []
for folder in sorted(glob.glob(os.path.join(base, "*"))):
    if not os.path.isdir(folder):
        continue
    graphs = os.path.join(folder, "run_simulation_graphs.txt")
    if not os.path.exists(graphs):
        continue
    js = os.path.join(folder, "job_script.sh")
    cores = None
    if os.path.exists(js):
        m = re.search(r"^#SBATCH -n (\d+)", open(js).read(), flags=re.M)
        cores = int(m.group(1)) if m else None
    data = [l.split() for l in open(graphs) if l.strip() and not l.startswith("#")]
    nsteps = len(data)
    t_last = float(data[-1][0]) if data else 0.0
    x_ct = float(data[-1][3]) if data else float("nan")
    wsteg = None
    for line in open(os.path.join(folder, "parameters.txt")):
        if line.startswith("wsteg="):
            wsteg = float(line.split("=")[1])
    wt, src = wall_time(folder)
    rows.append((os.path.basename(folder), cores, wsteg, nsteps, t_last, x_ct, wt, src))

print(f"{'case':38s} {'cores':>5s} {'w_s':>6s} {'steps':>7s} {'t_end':>8s} {'x_ct':>8s} {'wall[h]':>8s} {'s/step':>7s} {'full run est.[h]':>16s}  source")
for name, cores, wsteg, nsteps, t_last, x_ct, wt, src in rows:
    s_per_step = wt / nsteps if (wt and nsteps) else float("nan")
    est = s_per_step * STEPS_V2.get(wsteg, float("nan")) / 3600.0 if wt else float("nan")
    print(f"{name:38s} {str(cores):>5s} {wsteg:6.3f} {nsteps:7d} {t_last:8.3f} {x_ct:8.3f} {(wt or 0) / 3600:8.2f} {s_per_step:7.2f} {est:16.1f}  {src}")
print("\nfull-run estimate = s/step of the test x number of accepted steps of the corresponding v2.0 run (32 cores):")
print("   " + ", ".join(f"w_s={k}: {v}" for k, v in STEPS_V2.items()))
print("note: steps in the test are the cheap early steps (no crack jumps yet); real runs have more tiny steps around jumps -> add ~30 %.")
