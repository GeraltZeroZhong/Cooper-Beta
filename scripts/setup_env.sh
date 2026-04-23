#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/environment.yml"
ENV_NAME="cooperbeta"
WITH_DEV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_env.sh [--name ENV_NAME] [--dev] [--dry-run]

What it does:
  1. Prefer mamba/micromamba/conda to create an environment from environment.yml
     so DSSP is installed automatically.
  2. Install this repository in editable mode inside that environment.
  3. If no conda-like tool is available but apt-get exists, fall back to:
       - install DSSP from apt
       - create .venv
       - pip install the project

Options:
  --name ENV_NAME  Override the conda environment name (default: cooperbeta)
  --dev            Also install the project's dev dependencies
  --dry-run        Print commands without executing them
  -h, --help       Show this help message
EOF
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

choose_conda_frontend() {
  local candidate
  for candidate in mamba micromamba conda; do
    if have_cmd "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

conda_env_exists() {
  local frontend="$1"
  local env_name="$2"
  "$frontend" env list 2>/dev/null | awk 'NF && $1 !~ /^#/ { print $1 }' | grep -Fxq "$env_name"
}

install_spec() {
  if [[ "$WITH_DEV" -eq 1 ]]; then
    echo "$ROOT_DIR[full,dev]"
  else
    echo "$ROOT_DIR[full]"
  fi
}

setup_with_conda() {
  local frontend="$1"

  if conda_env_exists "$frontend" "$ENV_NAME"; then
    run_cmd "$frontend" env update --yes --name "$ENV_NAME" --file "$ENV_FILE" --prune
  else
    run_cmd "$frontend" env create --yes --name "$ENV_NAME" --file "$ENV_FILE"
  fi

  run_cmd "$frontend" run -n "$ENV_NAME" python -m pip install -e "$(install_spec)"

  echo
  echo "Environment is ready."
  if [[ "$frontend" == "micromamba" ]]; then
    echo "Activate it with: micromamba activate $ENV_NAME"
  else
    echo "Activate it with: conda activate $ENV_NAME"
  fi
}

setup_with_apt_venv() {
  local venv_dir="$ROOT_DIR/.venv"

  if [[ "$EUID" -ne 0 ]]; then
    run_cmd sudo apt-get update
    run_cmd sudo apt-get install -y dssp
  else
    run_cmd apt-get update
    run_cmd apt-get install -y dssp
  fi

  run_cmd python3 -m venv "$venv_dir"
  run_cmd "$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
  run_cmd "$venv_dir/bin/python" -m pip install -e "$(install_spec)"

  echo
  echo "Environment is ready."
  echo "Activate it with: source $venv_dir/bin/activate"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --dev)
      WITH_DEV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 1
fi

if frontend="$(choose_conda_frontend)"; then
  setup_with_conda "$frontend"
  exit 0
fi

if have_cmd apt-get; then
  setup_with_apt_venv
  exit 0
fi

cat >&2 <<'EOF'
No supported installer was found.

Recommended options:
  1. Install mamba/conda, then rerun: bash scripts/setup_env.sh
  2. Install DSSP manually and then run:
       python3 -m venv .venv
       source .venv/bin/activate
       pip install -e ".[full]"

You can also point Cooper-Beta to a custom DSSP binary with
`cooper_beta.config.Config.DSSP_BIN_PATH`.
EOF
exit 1
