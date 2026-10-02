# Continues the L2A stretch plan after the episodic linear-head pipeline finishes.
# Avoids PowerShell ErrorActionPreference=Stop on Python tqdm stderr.
$ErrorActionPreference = 'Continue'
$env:PYTHONPATH = 'src'
$env:PYTHONUNBUFFERED = '1'
$log = 'artifacts\l2a_stretch_remaining.log'
Set-Location 'd:\Proiecte\AgricultureAi'

function W([string]$m) {
    $line = '[{0}] {1}' -f (Get-Date -Format o), $m
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Output $line
}

function Invoke-Py([string[]]$argv) {
    W ('RUN ' + ($argv -join ' '))
    & .venv\Scripts\python.exe -u @argv >> $log 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed exit=" + $LASTEXITCODE + ' :: ' + ($argv -join ' '))
    }
}

W 'Phase A: wait for episodic linear-head pipeline ...'
while (-not ((Test-Path 'artifacts\sweep_m_area_l2a_episodic.log') -and ((Get-Content 'artifacts\sweep_m_area_l2a_episodic.log' -Raw -ErrorAction SilentlyContinue) -match 'Saved:'))) {
    Start-Sleep -Seconds 180
    $t = Get-Content 'artifacts\l2a_l3_episodic_pipeline.log' -Tail 1 -ErrorAction SilentlyContinue
    W ('waiting episodic: ' + $t)
}
W 'Episodic pipeline complete.'

W 'Phase B: Mahalanobis re-patch + 8-d metadata + frontier sweep'
Invoke-Py @(
    'scripts\patch_l3_distances.py',
    'artifacts\cached_logits_train.pt', 'artifacts\cached_logits_eval.pt',
    '--mode', 'mahalanobis',
    '--l3-weights', 'artifacts\checkpoints\level3_protonet_l2a_episodic\best.pt',
    '--level', 'L2A', '--batch-size', '256'
)
Invoke-Py @(
    'scripts\append_parcel_metadata.py',
    'artifacts\cached_logits_train.pt', 'artifacts\cached_logits_eval.pt',
    '--level', 'L2A', '--with-area', '--with-doy'
)
Invoke-Py @(
    'scripts\sweep_meta_frontier.py',
    '--multipliers=15', '--refresh-proba-cache',
    '--output-log', 'artifacts\sweep_meta_frontier_mahal.log',
    '--output-json', 'artifacts\sweep_meta_frontier_mahal.json',
    '--proba-cache', 'artifacts\meta_proba_cache_mahal.npz'
)
W 'REPATCH_AND_SWEEP_COMPLETE'

W 'Phase C: train DOY-aware L2A Transformer (30 epochs)'
Invoke-Py @(
    'scripts\train_transformer_breizhcrops.py',
    '--level', 'L2A', '--from-scratch', '--no-freeze-backbone', '--use-doy',
    '--epochs', '30', '--batch-size', '128',
    '--checkpoint-dir', 'artifacts\checkpoints\transformer_l2a_doy'
)
Copy-Item 'artifacts\checkpoints\transformer_l2a_doy\best.pt' 'artifacts\checkpoints\transformer\best.pt' -Force
W 'DOY_TRANSFORMER_COMPLETE'

W 'Phase D: rebuild caches'
Invoke-Py @(
    'scripts\build_meta_caches.py',
    '--l1-weights', 'artifacts\checkpoints\transformer_l2a_doy\best.pt',
    '--l3-weights', 'artifacts\checkpoints\level3_protonet_l2a_episodic\best.pt'
)

W 'Phase E: frontier sweep + ablation'
Invoke-Py @(
    'scripts\sweep_meta_frontier.py',
    '--multipliers=15', '--refresh-proba-cache',
    '--output-log', 'artifacts\sweep_meta_frontier_stretch.log',
    '--output-json', 'artifacts\sweep_meta_frontier_stretch.json',
    '--proba-cache', 'artifacts\meta_proba_cache_stretch.npz'
)
Invoke-Py @(
    'scripts\ablate_meta_features.py',
    '--output-log', 'artifacts\ablate_meta_features.log',
    '--output-json', 'artifacts\ablate_meta_features.json'
)

W 'L2A_STRETCH_PLAN_COMPLETE'
Write-Output 'AGENT_LOOP_WAKE_l2a_plan {"prompt":"All L2A stretch plan todos finished. Summarize results from mahal/stretch frontier and ablation logs."}'
