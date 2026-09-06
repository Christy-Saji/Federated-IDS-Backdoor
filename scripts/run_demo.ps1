# One-shot demo pipeline: real CIC-IDS2017 data, clean baseline vs BadNets
# attack, plus detection/defense numbers. Takes ~45-70 min end to end.
#
#   .\scripts\run_demo.ps1
#
# Everything is logged to results\demo_run.log and a plain-text summary is
# printed (and written to results\demo_summary.txt) at the end. Safe to
# re-run: the runner refuses to overwrite an existing run_id and just
# reports its cached summary instead.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$log = "results\demo_run.log"
New-Item -ItemType Directory -Force -Path results | Out-Null
"" | Set-Content $log

function Step($msg, $cmd) {
    Write-Host "`n=== $msg ===" -ForegroundColor Cyan
    Add-Content $log "`n=== $msg ==="
    & $cmd 2>&1 | Tee-Object -Append -FilePath $log
}

Step "1/5 clean-ASR real-data baseline (3 seeds x 20 rounds)" {
    & $py -m scripts.baselines.clean_asr --processed --seeds 0 1 2
}
Step "2/5 clean FedAvg, no attack, real data" {
    & $py -m flids.runner --config configs/clean_fedavg_real.yaml
}
Step "3/5 BadNets @ oob_999, no defense, real data" {
    & $py -m flids.runner --config configs/badnets_oob999_real.yaml
}
Step "4/5 detection AUC per aggregator (FLTrust vs FLAME vs FedAvg)" {
    & $py -m scripts.baselines.detection_auc --processed
}
Step "5/5 activation clustering TPR/FPR across poison ratios" {
    & $py -m scripts.baselines.activation_clustering --processed
}

Write-Host "`n=== DONE - building summary ===" -ForegroundColor Green

$runIds = Select-String -Path $log -Pattern '\[runner\] wrote results/([0-9a-f]+)/' |
    ForEach-Object { $_.Matches[0].Groups[1].Value }

$summary = New-Object System.Text.StringBuilder
[void]$summary.AppendLine("=== DEMO RESULTS ($(Get-Date)) ===`n")

foreach ($rid in $runIds) {
    $sf = "results\$rid\summary.json"
    if (Test-Path $sf) {
        [void]$summary.AppendLine("run_id $rid ->")
        [void]$summary.AppendLine((Get-Content $sf -Raw))
        [void]$summary.AppendLine("")
    }
}

[void]$summary.AppendLine("--- clean_asr.csv (real data) ---")
[void]$summary.AppendLine((Get-Content results\baselines\clean_asr.csv -Raw))
[void]$summary.AppendLine("--- detection_auc.csv ---")
[void]$summary.AppendLine((Get-Content results\baselines\detection_auc.csv -Raw))
[void]$summary.AppendLine("--- activation_clustering.csv ---")
[void]$summary.AppendLine((Get-Content results\baselines\activation_clustering.csv -Raw))

$summary.ToString() | Set-Content results\demo_summary.txt
Get-Content results\demo_summary.txt
Write-Host "`nFull log: $log" -ForegroundColor Yellow
Write-Host "Summary:  results\demo_summary.txt" -ForegroundColor Yellow
