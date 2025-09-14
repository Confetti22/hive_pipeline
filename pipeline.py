# README (quickstart)
# -------------------
# 1) Put your pipeline.yaml somewhere (e.g., configs/pipeline.yaml) — you already have it.
# 2) Run end‑to‑end locally:
#    python pipeline.py -cfg configs/pipeline.yaml
# 3) Run a single step (with dependency checks):
#    python pipeline.py -cfg configs/pipeline.yaml -step crop
#    python pipeline.py -cfg configs/pipeline.yaml -step train_ae
#    python pipeline.py -cfg configs/pipeline.yaml -step extract
#    python pipeline.py -cfg configs/pipeline.yaml -step train_mlp
# 4) Slurm mode (submit each step via sbatch; uses exec.sbatch in YAML):
#    python pipeline.py -cfg configs/pipeline.yaml -use_slurm
# 5) Utilities:
#    -list_steps   : print available steps
#    -force        : re-run a step even if its outputs exist
#    -dry          : print what would run without executing
#
# Directory layout (automatically created):
#   runs/<run_id>/
#     data/<ae_train_folder>/   (cropped train ROIs)
#     data/<ae_test_folder>/    (cropped test  ROIs)
#     data/ae_feats.zarr     (features written by extract step)
#     <ae_out_folder>/          (AE ckpts, logs)
#     <contrastive_out_folder>  (MLP ckpts, logs)
#
# NOTE about your YAML: feature_extract has `batch_size` listed twice. YAML keeps the LAST one (1024).
# This orchestrator logs a warning when it detects duplicate keys.


# ---------------------------
# pipeline.py (orchestrator)
# ---------------------------
import argparse
import os
import sys
import subprocess
import textwrap
from pathlib import Path

# --- local utils (defined below in this single file for portability) ---
import yaml
from collections import defaultdict


# config_loader.py
from collections import defaultdict
import yaml

def _load_cfg(path: str, resolve: bool = True):
    """
    Load YAML with:
      - duplicate-key warnings (PyYAML pre-pass),
      - OmegaConf interpolation support (real load),
      - optional full resolution to a plain dict.
    """
    # --- 1) Duplicate-key warning pre-pass (no return value used) ---
    class DupesLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        seen = defaultdict(list)
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                # +1 to convert 0-based to 1-based line numbers
                seen[key].append(value_node.start_mark.line + 1)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        if seen:
            print("[WARN] Duplicate keys detected in YAML:")
            for k, lines in seen.items():
                print(f"       key='{k}' occurs multiple times at lines {lines} — last one wins.")
        return mapping

    DupesLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    # Run the warning pass
    with open(path, "r") as f:
        yaml.load(f, Loader=DupesLoader)

    # --- 2) Real load with OmegaConf for ${...} interpolation ---
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise RuntimeError(
            "OmegaConf is required for interpolation. Install with: pip install omegaconf"
        ) from e

    cfg = OmegaConf.load(path)

    # --- 3) Optionally resolve to a plain dict (easy to pass around) ---
    if resolve:
        # resolve=True makes ${...} substitutions concrete
        return OmegaConf.to_container(cfg, resolve=True)
    else:
        # Return the OmegaConf object if you want to keep lazy interpolation
        return cfg


def load_cfg(cfg_path: str) -> dict:
    cfg = _load_cfg(cfg_path)
    # expand user and make some derived paths
    run_id = cfg['run_id']
    root = Path(cfg['paths']['output_root']).expanduser().absolute() / run_id

    paths = cfg['paths']
    cfg['_run'] = {
        'dims': cfg['dims'],
        'input_image': Path(cfg['paths']['input_image']).expanduser().absolute(),
        'root': root,
        'run_id': run_id,
        'data_dir': root / 'data',
        'ae_train_folder': root / 'data' / paths['ae_train_folder'],
        'ae_test_folder': root / 'data' / paths['ae_test_folder'],
        'ae_out_folder': root / paths['ae_out_folder'],
        'contrastive_out_folder': root / paths['contrastive_out_folder'],
        'zarr_path': root / 'data' / cfg['paths']['zarr_name'],
    }
    return cfg


def ensure_dirs(cfg: dict):
    for p in cfg['_run'].values():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)
    cfg['_run']['root'].mkdir(parents=True, exist_ok=True)
    cfg['_run']['data_dir'].mkdir(parents=True, exist_ok=True)


def step_already_done(step: str, cfg: dict) -> bool:
    R = cfg['_run']
    if step == 'crop':
        return R['ae_train_folder'].exists() and any(R['ae_train_folder'].glob('*.tif')) and R['ae_test_folder'].exists() and any(R['ae_test_folder'].glob('*.tif'))
    if step == 'train_ae':
        ckpt_path = Path(R['ae_out_folder'] / 'weights' /cfg['autoencoder']["ae_exp_name"]/'best.pth')
        return ckpt_path.exists()
    if step == 'extract':
        return Path(R['zarr_path']).exists()
    if step == 'train_mlp':
        ckpt_path = R['contrastive_out_folder'] / 'weights' / cfg['contrastive_mlp']['contastive_exp_name']/'best.pth'

        return ckpt_path.exists()
    return False


def dep_checks(step: str, cfg: dict):
    R = cfg['_run']
    if step == 'train_ae':
        if not (R['ae_train_folder'].exists() and any(R['ae_train_folder'].glob('*.tif'))):
            raise SystemExit("[ERR] Missing cropped train ROIs — run step 'crop' first.")
        if not (R['ae_test_folder'].exists() and any(R['ae_test_folder'].glob('*.tif'))):
            raise SystemExit("[ERR] Missing cropped test ROIs — run step 'crop' first.")
    if step == 'extract':
        ckpt = Path(cfg['paths']['ae_weight_path'])
        if not ckpt.exists():
            raise SystemExit("[ERR] Missing AE checkpoint — run step 'train_ae' first.")
    if step == 'train_mlp':
        if not Path(R['zarr_path']).exists():
            raise SystemExit("[ERR] Missing feature zarr — run step 'extract' first.")


def build_env(cfg: dict) -> dict:
    env = os.environ.copy()
    env['PIPELINE_CFG'] = str(cfg)
    return env


def build_cmd(step: str, cfg_path: str) -> list:
    mapping = {
        'crop': 'scripts/crop_dataset.py',
        'train_ae': 'scripts/train_autoencoder.py',
        'extract': 'scripts/extract_feats.py',
        'train_mlp': 'scripts/train_contrastive.py',
    }
    if step not in mapping:
        raise SystemExit(f"Unknown step: {step}")
    return [sys.executable, mapping[step], '-cfg', cfg_path]


def run_local(cmd: list, dry: bool):
    print('[CMD]', ' '.join(map(str, cmd)))
    if dry:
        return 0
    return subprocess.call(cmd)


def run_slurm(cmd: list, cfg: dict, dry: bool):
    sb = cfg['exec'].get('sbatch', {})
    run_id = cfg['run_id']
    job_name = f"{run_id}-{Path(cmd[1]).stem}"
    script = textwrap.dedent(f"""
    #!/bin/sh
    #SBATCH --job-name={job_name}
    #SBATCH --partition={sb.get('partition','compute')}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={sb.get('cpus_per_task',8)}
    #SBATCH --mem={sb.get('mem','20G')}
    #SBATCH --time={sb.get('time','24:00:00')}
    {f"#SBATCH --nodelist={sb.get('nodelist')}" if sb.get('nodelist') else ''}
    {f"#SBATCH --gres=gpu:{sb.get('gpus',1)}" if sb.get('gpus',0) else ''}
    {sb.get('extra_args','')}

    echo "[SLURM] Running: {' '.join(map(str, cmd))}"
    {' '.join(map(str, cmd))}
    """)
    slurm_sh = Path('runs') / f"{job_name}.sbatch.sh"
    slurm_sh.parent.mkdir(parents=True, exist_ok=True)
    slurm_sh.write_text(script)
    print('[SBATCH SCRIPT]', slurm_sh)
    if dry:
        print(script)
        return 0
    return subprocess.call(['sbatch', str(slurm_sh)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', default='config/pipeline.yaml',help='Path to pipeline.yaml')
    steps_all = ['crop','train_ae','extract','train_mlp']
    parser.add_argument('-step', choices=steps_all, help='Run only one step')
    parser.add_argument('-force', action='store_true', help='Re-run even if outputs exist')
    parser.add_argument('-dry', action='store_true', help='Print commands without executing')
    parser.add_argument('-list_steps', action='store_true', help='List available steps')
    parser.add_argument('-use_slurm', action='store_true', help='Override YAML and submit via sbatch')
    args = parser.parse_args()

    if args.list_steps:
        print('Available steps: ' + ', '.join(steps_all))
        sys.exit(0)

    cfg = load_cfg(args.cfg)
    ensure_dirs(cfg)

    steps = steps_all if args.step is None else [args.step]
    use_slurm = args.use_slurm or cfg.get('exec',{}).get('use_slurm', False)

    for step in steps:
        #after training ae and mlp, the yaml is updated, reload it before running next step
        cfg = load_cfg(args.cfg)

        if not args.force and step_already_done(step, cfg):
            print(f"[SKIP] {step} is already done. Use -force to re-run.")
            continue
        if step != 'crop':
            dep_checks(step, cfg)
        cmd = build_cmd(step, args.cfg)
        rc = run_slurm(cmd, cfg, args.dry) if use_slurm else run_local(cmd, args.dry)
        if rc != 0:
            raise SystemExit(f"[ERR] Step '{step}' failed with return code {rc}")

if __name__ == '__main__':
    main()
