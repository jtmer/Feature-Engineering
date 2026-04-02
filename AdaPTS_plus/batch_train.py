# batch_train.py
import os
import time
import json
import shlex
import signal
import subprocess
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple


# =========================
# User settings
# =========================
SKIP_COMPLETED = True

PYTHON_EXEC = "python"
MAIN_FILE = "main.py"

# 你机器上可用的物理 GPU id
GPU_IDS = list(range(8))

# 轮询间隔（秒）
POLL_INTERVAL = 30

# 判定“空闲卡”的阈值
MAX_USED_MEM_MB = 1500
MIN_FREE_UTIL = 10  # util.gpu <= 10 认为够空闲

# 并发上限（通常 <= 8）
MAX_CONCURRENT_JOBS = 8

# 基础命令参数：这里放你不打算 sweep 的固定参数
BASE_ARGS = {
    "--data_path": "/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv",
    "--out_root": "./results/results_monthly_backtest",
    "--time_col": "time",
    "--target_col": "day_ahead_clearing_price",
    "--drop_cols": "time,realtime_clearing_price",
    "--start_ym": "2024-08",
    "--end_ym": "2025-07",
    "--train_years": 2,
    "--val_months": 1,
    "--target_hour": 0,
    "--freq_min": 15,
    "--hist_window": 2880,
    "--pred_window": 96,
    "--stride_train": 96,
    "--stride_eval": 96,
    "--seed": 13,
    "--model_name": "thuml/sundial-base-128m",
    "--z_dim": 1,
    "--patch_size": 16,
    "--encoder_type": "neural",
    "--decoder_type": "neural",
    "--normalizer": "identity",
    "--stats_pretrain_epochs": 100,
    "--stats_pretrain_bs": 64,
    "--stats_pretrain_lr": 1e-4,
    "--stats_pretrain_wd": 1e-4,
    "--stats_pretrain_patience": 8,
    "--past_pretrain_epochs": 200,
    "--past_pretrain_min_epochs": 60,
    "--past_pretrain_bs": 32,
    "--past_pretrain_lr": 5e-4,
    "--past_pretrain_wd": 1e-4,
    "--past_pretrain_patience": 15,
    "--train_epochs": 60,
    "--train_bs": 32,
    "--train_wd": 1e-4,
    "--lambda_latent_stats": 0.5,
    "--ltm_batch_size_train": 1,
    "--ltm_batch_size_pred": 32,
    "--n_samples": 30,
    "--max_vis_windows": 6,
    "--freeze_decoder": 0,
    "--encoder_lr": 1e-4,
    "--stats_lr": 1e-4,
    "--debug": 1,
    "--debug_every": 2,
    "--debug_num_seq": 2,
    "--debug_latent_ch": 1,
    "--earlystop_patience": 10,
    "--auto_out_root": 1,
    "--num_workers": 2,
    "--pin_memory": 1,
    "--persistent_workers": 1,
    "--prefetch_factor": 2,
}

# =========================
# 这里定义要 sweep 的超参数
# 注意：全排列会非常多，建议先少量组合验证
# =========================
# lambda_latent_scale、latent_rms_target、 train_lr、
# lambda_past_recon、lambda_future_pred、lambda_stats_pred、lambda_y_patch_std、
# lambda_proxy_floor、proxy_sens_floor、lambda_ratio_floor、proxy_cov_ratio_floor、lambda_monitor、latent_std_floor、lambda_latent_std_floor、
# decoder_lr
SEARCH_SPACE = {
    "--lambda_latent_scale": [1e-4, 0],
    "--latent_rms_target": [10.0, 25.0],
    "--train_lr": [5e-5],
    "--lambda_past_recon": [0.5, 1.0],
    "--lambda_future_pred": [2.0, 4.0],
    "--lambda_stats_pred": [0.0, 0.2],
    "--lambda_y_patch_std": [0.5, 1.0],
    "--lambda_proxy_floor": [2.0],
    "--proxy_sens_floor": [0.40],
    "--lambda_ratio_floor": [1.0],
    "--proxy_cov_ratio_floor": [0.40],
    "--lambda_monitor": [1.0],
    "--latent_std_floor": [0.3],
    "--lambda_latent_std_floor": [0.0],
    "--decoder_lr": [1e-4, 5e-5, 1e-5],
}


def format_hp_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.8g}".replace("-", "m").replace(".", "p")
    return str(v).replace("/", "-").replace(" ", "")


def build_exp_suffix(cfg: Dict[str, Any]) -> str:
    alias = {
        "--lambda_latent_scale": "lls",
        "--latent_rms_target": "lrt",
        "--train_lr": "tlr",
        "--lambda_past_recon": "lpr",
        "--lambda_future_pred": "lfp",
        "--lambda_stats_pred": "lsp",
        "--lambda_y_patch_std": "lyps",
        "--lambda_proxy_floor": "lpf",
        "--proxy_sens_floor": "psf",
        "--lambda_ratio_floor": "lrf",
        "--proxy_cov_ratio_floor": "pcf",
        "--lambda_monitor": "lmon",
        "--latent_std_floor": "lsf",
        "--lambda_latent_std_floor": "llsf",
        "--decoder_lr": "dlr",
    }
    parts = []
    for k in SEARCH_SPACE.keys():
        parts.append(f"{alias[k]}{format_hp_value(cfg[k])}")
    return "_".join(parts)


def generate_jobs() -> List[Dict[str, Any]]:
    keys = list(SEARCH_SPACE.keys())
    value_lists = [SEARCH_SPACE[k] for k in keys]
    jobs = []
    for values in itertools.product(*value_lists):
        one = dict(BASE_ARGS)
        for k, v in zip(keys, values):
            one[k] = v
        one["--exp_suffix"] = build_exp_suffix(one)
        jobs.append(one)
    return jobs

def is_job_completed(job_cfg: Dict[str, Any]) -> bool:
    exp_suffix = job_cfg["--exp_suffix"]
    out_dir = Path(job_cfg["--out_root"] + "_" + exp_suffix)

    # 你也可以换成别的更严格标记
    done_file = out_dir / "results_summary.csv"
    return done_file.exists()

def query_gpu_status() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = []
    for line in out.strip().splitlines():
        idx_str, mem_str, util_str = [x.strip() for x in line.split(",")]
        idx = int(idx_str)
        if idx not in GPU_IDS:
            continue
        rows.append({
            "index": idx,
            "memory_used": int(mem_str),
            "util": int(util_str),
        })
    return rows


def find_free_gpus() -> List[int]:
    rows = query_gpu_status()
    free_ids = []
    for r in rows:
        if r["memory_used"] <= MAX_USED_MEM_MB and r["util"] <= MIN_FREE_UTIL:
            free_ids.append(r["index"])
    return sorted(free_ids)


def build_cmd(args_dict: Dict[str, Any]) -> List[str]:
    cmd = [PYTHON_EXEC, MAIN_FILE]
    for k, v in args_dict.items():
        cmd.append(k)
        cmd.append(str(v))
    return cmd


def launch_job(job_cfg: Dict[str, Any], gpu_id: int, work_dir: str = ".") -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    exp_suffix = job_cfg["--exp_suffix"]
    launcher_dir = Path(job_cfg["--out_root"] + "_" + exp_suffix)
    launcher_dir.mkdir(parents=True, exist_ok=True)

    with open(launcher_dir / "launcher_job_config.json", "w", encoding="utf-8") as f:
        json.dump(job_cfg, f, ensure_ascii=False, indent=2)

    stdout_f = open(launcher_dir / "launcher_stdout.log", "w", encoding="utf-8")
    stderr_f = open(launcher_dir / "launcher_stderr.log", "w", encoding="utf-8")

    cmd = build_cmd(job_cfg)
    print(f"[launch] gpu={gpu_id} exp={exp_suffix}")
    print("[launch] cmd:", " ".join(shlex.quote(x) for x in cmd))

    p = subprocess.Popen(
        cmd,
        cwd=work_dir,
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        preexec_fn=os.setsid,
    )
    p._stdout_f = stdout_f
    p._stderr_f = stderr_f
    p._stdout_path = str(launcher_dir / "launcher_stdout.log")
    p._stderr_path = str(launcher_dir / "launcher_stderr.log")
    p._gpu_id = gpu_id
    p._exp_suffix = exp_suffix
    p._job_cfg = job_cfg
    p._out_dir = str(launcher_dir)
    return p

def record_failed_job(job_cfg, gpu_id, returncode, failed_log_path, out_dir=None, stderr_path=None):
    rec = {
        "gpu_id": gpu_id,
        "returncode": returncode,
        "job_cfg": job_cfg,
        "out_dir": out_dir,
        "stderr_path": stderr_path,
    }
    with open(failed_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")



FAILED_LOG_PATH = "batch_failed_jobs.jsonl"

def cleanup_finished(running: List[subprocess.Popen]) -> Tuple[List[subprocess.Popen], List[subprocess.Popen]]:
    alive = []
    finished = []

    for p in running:
        ret = p.poll()
        if ret is None:
            alive.append(p)
            continue

        finished.append(p)

        try:
            p._stdout_f.close()
            p._stderr_f.close()
        except Exception:
            pass

        if ret != 0:
            print(f"[error] gpu={p._gpu_id} exp={p._exp_suffix} returncode={ret}")
            record_failed_job(
                job_cfg=p._job_cfg,
                gpu_id=p._gpu_id,
                returncode=ret,
                failed_log_path=FAILED_LOG_PATH,
                out_dir=p._out_dir,
                stderr_path=p._stderr_path,
            )
        else:
            print(f"[done] gpu={p._gpu_id} exp={p._exp_suffix} returncode={ret}")

    return alive, finished


def main():
    jobs = generate_jobs()
    print(f"[info] total jobs = {len(jobs)}")

    if SKIP_COMPLETED:
        completed_jobs = [job for job in jobs if is_job_completed(job)]
        pending = [job for job in jobs if not is_job_completed(job)]
        print(f"[info] completed jobs found = {len(completed_jobs)}")
        print(f"[info] pending jobs to run   = {len(pending)}")
    else:
        completed_jobs = []
        pending = jobs[:]
        print("[info] SKIP_COMPLETED=False, run all jobs")
    
    running: List[subprocess.Popen] = []

    try:
        while pending or running:
            running, _ = cleanup_finished(running)

            current_free = find_free_gpus()
            occupied_by_us = {p._gpu_id for p in running if p.poll() is None}
            launchable_gpus = [g for g in current_free if g not in occupied_by_us]

            while pending and launchable_gpus and len(running) < MAX_CONCURRENT_JOBS:
                gpu_id = launchable_gpus.pop(0)
                job_cfg = pending.pop(0)
                p = launch_job(job_cfg, gpu_id=gpu_id, work_dir=".")
                running.append(p)

            print(
                f"[loop] pending={len(pending)} running={len(running)} "
                f"free_gpus={current_free}"
            )

            if pending or running:
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("[warn] KeyboardInterrupt, terminate all running jobs ...")
        for p in running:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()