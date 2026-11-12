"""Run the whole RLHF pipeline on Modal.
  smoke test : python -m modal run modal_app.py --mode smoke
  full run   : python -m modal run --detach modal_app.py --mode full
  progress   : python -m modal volume get rlhf-vol full/progress.log -
"""
import modal, pathlib

REPO = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers", "datasets", "rouge-score", "matplotlib", "numpy")
         .env({"HF_HOME": "/vol/hf"})
         .add_local_dir(REPO, "/src", ignore=["checkpoints", ".git", "*.ipynb", "__pycache__"]))
vol = modal.Volume.from_name("rlhf-vol", create_if_missing=True)
app = modal.App("rlhf-summarize", image=image)

STAGES = ["common.py", "train_sft.py", "train_reward.py", "train_ppo.py", "train_dpo.py", "evaluate.py"]

SMOKE_SUBS = [  # (file, regex, replacement)
    ("starter/config.py", r"N_SFT_TRAIN, N_SFT_VAL\s*=.*", "N_SFT_TRAIN, N_SFT_VAL = 64, 16"),
    ("starter/config.py", r"N_PREF_TRAIN\s*,\s*N_PREF_VAL\s*=.*", "N_PREF_TRAIN, N_PREF_VAL = 64, 16"),
    ("starter/config.py", r"N_EVAL_POSTS\s*=.*", "N_EVAL_POSTS = 8"),
    ("train_ppo.py", r"ITERATIONS = 4, 1e-5, 16, 200", "ITERATIONS = 4, 1e-5, 16, 3"),
    ("train_sft.py", r"LOG_EVERY = .*", "LOG_EVERY = 5e-5, 16, 2"),
    ("train_reward.py", r"LOG_EVERY = .*", "LOG_EVERY = 1e-5, 8, 2"),
    ("train_dpo.py", r"LOG_EVERY = .*", "LOG_EVERY = 0.1, 2e-6, 8, 2"),
]


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=24 * 3600)
def run(mode: str = "smoke", start: int = 0):
    import os, re, shutil, subprocess, time
    work = f"/vol/{mode}"
    shutil.copytree("/src", work, dirs_exist_ok=True)
    os.makedirs(f"{work}/results", exist_ok=True)
    os.makedirs(f"{work}/checkpoints", exist_ok=True)
    if mode == "smoke":
        for f, pat, rep in SMOKE_SUBS:
            p = f"{work}/{f}"
            txt = open(p).read()
            new = re.sub(pat, rep, txt, count=1)
            open(p, "w").write(new)
    # results/table2 is appended to; reset it on a fresh run only
    if start == 0: open(f"{work}/results/table2_training.md", "w").write(
        "| Stage | Final train loss | Validation metric | Time / GPU |\n|---|---|---|---|\n")
    plog = open(f"{work}/progress.log", "a")
    def note(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True); plog.write(line + "\n"); plog.flush(); vol.commit()
    note(f"=== start mode={mode} ===")
    for s in STAGES[start:]:
        note(f"BEGIN {s}")
        t0 = time.time()
        with open(f"{work}/results/{s}.log", "w") as out:
            proc = subprocess.Popen(["python", "-u", s], cwd=work, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            last_commit = time.time()
            for line in proc.stdout:
                print(line, end="", flush=True); out.write(line)
                if time.time() - last_commit > 120:
                    out.flush(); vol.commit(); last_commit = time.time()
            rc = proc.wait()
        note(f"END {s} rc={rc} {time.time() - t0:.0f}s")
        if rc != 0:
            note("FAILED"); return
    note("ALL DONE")


@app.local_entrypoint()
def main(mode: str = "smoke", start: int = 0):
    run.remote(mode, start)
