#!/usr/bin/env bash
# Create the uv environment and (optionally) log in to Hugging Face.
#   bash setup.sh          # base environment
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
echo
echo "Llama-2 / Llama-3 are gated on Hugging Face. Accept their licences on the"
echo "model pages, then authenticate once with either:"
echo "    uv run huggingface-cli login        # or:  export HF_TOKEN=..."
echo "Next: uv run python scripts/download_models.py && uv run python scripts/download_data.py"
