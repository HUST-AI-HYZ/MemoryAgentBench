# Run MemoryAgentBench against the AgentMemoryToolkit (Cosmos DB) backend.
#
# Usage:
#   .\bash_files\sh\run_memagent_amt.ps1
#   .\bash_files\sh\run_memagent_amt.ps1 -AgentYaml Structure_rag_gpt-5.4-mini-amt.yaml `
#                                        -DatasetYaml Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml
#
# Defaults run the gpt-5.4-mini AMT agent against LongMemEval_s_star.

[CmdletBinding()]
param(
    [string]$AgentYaml = 'Structure_rag_gpt-5.4-mini-amt.yaml',
    [string]$DatasetYaml = 'Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml',
    [string]$AgentSubdir = 'gpt-5.4-mini'
)

$ErrorActionPreference = 'Stop'

# Activate venv if present and not already active.
if (-not $env:VIRTUAL_ENV) {
    $venvActivate = Join-Path $PWD '.venv\Scripts\Activate.ps1'
    if (Test-Path $venvActivate) {
        . $venvActivate
    }
}

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'

$agentConfigPath   = Join-Path $PWD "configs\agent_conf\RAG_Agents\$AgentSubdir\$AgentYaml"
$datasetConfigPath = Join-Path $PWD "configs\data_conf\$($DatasetYaml -replace '/', '\')"

Write-Host '................Start...........'
Write-Host "agent_config:   $agentConfigPath"
Write-Host "dataset_config: $datasetConfigPath"

python main.py `
    --agent_config   $agentConfigPath `
    --dataset_config $datasetConfigPath

Write-Host '................End...........'
