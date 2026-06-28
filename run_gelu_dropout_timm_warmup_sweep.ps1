# GELU/dropout timm staged-finetuning sweep:
#   global K in {1, 64, 1024}
#   patch-location K in {1, 64}
#   5 projection-head-only epochs, then the last 2 backbone blocks
#   are unfrozen at 1e-6 while the head uses 1e-4
#
# Compare these runs against the existing frozen+timm results from the original
# GELU/dropout sweep; the frozen controls do not need to be rerun.
#
# Usage:
#   .\run_gelu_dropout_timm_warmup_sweep.ps1 -ValidateOnly
#   .\run_gelu_dropout_timm_warmup_sweep.ps1 -SkipCompleted
#   .\run_gelu_dropout_timm_warmup_sweep.ps1 -SkipCompleted -StopOnFailure

param(
    [switch]$ValidateOnly,
    [switch]$SkipCompleted,
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"
$env:HF_HUB_OFFLINE = "1"
$env:LOKY_MAX_CPU_COUNT = "8"

$root = $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"
$main = Join-Path $root "project\main.py"
$runsRoot = Join-Path $root "runs"
$logDir = Join-Path $runsRoot "gelu_dropout_timm_warmup_sweep_logs"
$timingPath = Join-Path $runsRoot "gelu_dropout_timm_warmup_sweep_timings.json"

$experiments = @(
    @{ config = "project\configs\gelu_dropout_global_k1_finetune_timm.yaml";              name = "gelu_dropout_global_k1_finetune_timm" },
    @{ config = "project\configs\gelu_dropout_patch_location_k1_finetune_timm.yaml";      name = "gelu_dropout_patch_location_k1_finetune_timm" },

    @{ config = "project\configs\gelu_dropout_global_k64_finetune_timm.yaml";             name = "gelu_dropout_global_k64_finetune_timm" },
    @{ config = "project\configs\gelu_dropout_patch_location_k64_finetune_timm.yaml";     name = "gelu_dropout_patch_location_k64_finetune_timm" },

    @{ config = "project\configs\gelu_dropout_global_k1024_finetune_timm.yaml";           name = "gelu_dropout_global_k1024_finetune_timm" }
)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}
if (-not (Test-Path -LiteralPath $main)) {
    throw "Training entrypoint not found: $main"
}
foreach ($experiment in $experiments) {
    $configPath = Join-Path $root $experiment.config
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Sweep config not found: $configPath"
    }
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$runStartedAt = Get-Date
$timings = [System.Collections.ArrayList]::new()
$hadFailures = $false

function Save-TimingSummary {
    param(
        [datetime]$StartedAt,
        [System.Collections.ArrayList]$Entries,
        [string]$Path,
        [string]$Status
    )

    $finishedAt = Get-Date
    $summary = [ordered]@{
        started_at = $StartedAt.ToString("o")
        finished_at = $finishedAt.ToString("o")
        total_minutes = [math]::Round(($finishedAt - $StartedAt).TotalMinutes, 2)
        status = $Status
        experiments = $Entries
    }

    $json = $summary | ConvertTo-Json -Depth 6
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

function Test-ExperimentCompleted {
    param([string]$ExperimentName)

    $outerDir = Join-Path $runsRoot $ExperimentName
    $candidateDirs = @($outerDir)
    if (Test-Path -LiteralPath $outerDir) {
        $candidateDirs += @(
            Get-ChildItem -LiteralPath $outerDir -Directory |
                Where-Object { $_.Name -eq $ExperimentName -or $_.Name -like "$ExperimentName`_*" } |
                Select-Object -ExpandProperty FullName
        )
    }

    foreach ($experimentDir in ($candidateDirs | Select-Object -Unique)) {
        $metricsPath = Join-Path $experimentDir "evaluation\evaluation_metrics.json"
        $stage2Model = Join-Path $experimentDir "final_stage2_model.pth"
        $stage1Model = Join-Path $experimentDir "final_model.pth"
        if (
            (Test-Path -LiteralPath $metricsPath) -and
            ((Test-Path -LiteralPath $stage2Model) -or (Test-Path -LiteralPath $stage1Model))
        ) {
            return $true
        }
    }

    return $false
}

if ($ValidateOnly) {
    Write-Host "Validated $($experiments.Count) sweep entries:" -ForegroundColor Green
    foreach ($experiment in $experiments) {
        $completed = Test-ExperimentCompleted -ExperimentName $experiment.name
        Write-Host "  $($experiment.name): completed=$completed"
    }
    exit 0
}

Set-Location $root

foreach ($experiment in $experiments) {
    $configPath = Join-Path $root $experiment.config
    $logPath = Join-Path $logDir "$($experiment.name).log"

    if ($SkipCompleted -and (Test-ExperimentCompleted -ExperimentName $experiment.name)) {
        $skippedAt = Get-Date
        [void]$timings.Add([ordered]@{
            name = $experiment.name
            config = $configPath
            log_path = $logPath
            started_at = $skippedAt.ToString("o")
            finished_at = $skippedAt.ToString("o")
            elapsed_minutes = 0.0
            total_elapsed_minutes = [math]::Round(($skippedAt - $runStartedAt).TotalMinutes, 2)
            exit_code = 0
            status = "skipped_completed"
        })
        Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "running"
        Write-Host "SKIP: $($experiment.name) already completed" -ForegroundColor Yellow
        continue
    }

    $temporarySuffix = [System.Guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path $logDir "$($experiment.name).$temporarySuffix.stdout.tmp"
    $stderrPath = Join-Path $logDir "$($experiment.name).$temporarySuffix.stderr.tmp"
    $experimentStartedAt = Get-Date

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Starting: $($experiment.name)" -ForegroundColor Cyan
    Write-Host "  Config: $configPath" -ForegroundColor DarkCyan
    Write-Host "  Started at: $($experimentStartedAt.ToString('u'))" -ForegroundColor DarkCyan
    Write-Host "================================================================" -ForegroundColor Cyan

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @($main, "--config", $configPath, "--exp-name", $experiment.name) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    $exitCode = $process.ExitCode

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($logPath, "STDOUT`r`n======`r`n", $encoding)
    if (Test-Path -LiteralPath $stdoutPath) {
        [System.IO.File]::AppendAllText($logPath, [System.IO.File]::ReadAllText($stdoutPath), $encoding)
    }
    [System.IO.File]::AppendAllText($logPath, "`r`nSTDERR`r`n======`r`n", $encoding)
    if (Test-Path -LiteralPath $stderrPath) {
        [System.IO.File]::AppendAllText($logPath, [System.IO.File]::ReadAllText($stderrPath), $encoding)
    }
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

    $experimentFinishedAt = Get-Date
    $elapsedMinutes = [math]::Round(($experimentFinishedAt - $experimentStartedAt).TotalMinutes, 2)
    $status = if ($exitCode -eq 0) { "completed" } else { "failed" }
    [void]$timings.Add([ordered]@{
        name = $experiment.name
        config = $configPath
        log_path = $logPath
        started_at = $experimentStartedAt.ToString("o")
        finished_at = $experimentFinishedAt.ToString("o")
        elapsed_minutes = $elapsedMinutes
        total_elapsed_minutes = [math]::Round(($experimentFinishedAt - $runStartedAt).TotalMinutes, 2)
        exit_code = $exitCode
        status = $status
    })
    Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "running"

    if ($exitCode -ne 0) {
        $hadFailures = $true
        Write-Host "FAILED: $($experiment.name) (exit $exitCode)" -ForegroundColor Red
        if ($StopOnFailure) {
            Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "failed"
            exit $exitCode
        }
        continue
    }

    Write-Host "DONE: $($experiment.name) ($elapsedMinutes min)" -ForegroundColor Green
}

$finalStatus = if ($hadFailures) { "completed_with_failures" } else { "completed" }
Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status $finalStatus

Write-Host ""
Write-Host "Sweep status: $finalStatus" -ForegroundColor Green
Write-Host "Timing log: $timingPath" -ForegroundColor Green
Write-Host "Logs: $logDir" -ForegroundColor Green

if ($hadFailures) {
    exit 1
}
