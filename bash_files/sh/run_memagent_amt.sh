#!/usr/bin/env bash
# Run MemoryAgentBench against the AgentMemoryToolkit (Cosmos DB) backend.
#
# Usage:
#   bash bash_files/sh/run_memagent_amt.sh
#   bash bash_files/sh/run_memagent_amt.sh <agent_yaml_basename> <dataset_yaml_relpath>
#
# Defaults run the gpt-5.4-mini AMT agent against LongMemEval_s_star.

set -euo pipefail

# Activate venv if present (skip if already in one).
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/Scripts/activate ]]; then
        source .venv/Scripts/activate
    elif [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    fi
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)

agent_yaml=${1:-Structure_rag_gpt-5.4-mini-amt.yaml}
dataset_yaml=${2:-Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml}

agent_config_path="${root}/configs/agent_conf/RAG_Agents/gpt-5.4-mini/${agent_yaml}"
dataset_config_path="${root}/configs/data_conf/${dataset_yaml}"

echo "................Start..........."
echo "agent_config:   ${agent_config_path}"
echo "dataset_config: ${dataset_config_path}"

python main.py \
    --agent_config   "${agent_config_path}" \
    --dataset_config "${dataset_config_path}"

echo "................End..........."

# Examples:
#   bash bash_files/sh/run_memagent_amt.sh
#   bash bash_files/sh/run_memagent_amt.sh Structure_rag_gpt-5.4-mini-amt.yaml \
#                                          Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml
