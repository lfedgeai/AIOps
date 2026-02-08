#!/usr/bin/env bash
set -euo pipefail

# This script installs Ollama and pulls three high-quality instruction-tuned models
# for local experimentation: Llama 3.1, Mistral, and Qwen2.5.
# It does not start any background service (Ollama runs as a daemon on first use).

echo "[+] Installing Ollama (see https://ollama.com for platform details)..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "[+] Ollama already installed."
fi

echo "[+] Pulling models (this may take a while):"
models=("llama3.1" "mistral" "qwen2.5")
for m in "${models[@]}"; do
  echo "    - $m"
  ollama pull "$m" || true
done

echo "[+] Set base URL for the harness (default Ollama port is 11434):"
echo "    export OLLAMA_BASE_URL=http://127.0.0.1:11434"
echo
echo "[+] Test the API:"
echo "    curl -s http://127.0.0.1:11434/api/tags | jq ."
echo
echo "[+] Run harness examples, e.g.:"
echo "    aiops-harness run --config examples/configs/ollama_llama3.yaml --dataset examples/data"
echo "    aiops-harness run --config examples/configs/ollama_mistral.yaml --dataset examples/data"
echo "    aiops-harness run --config examples/configs/ollama_qwen.yaml --dataset examples/data"

