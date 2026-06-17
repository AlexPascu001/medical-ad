# Run the GELU/dropout experiment sweep:
#   K in {1, 64, 1024}
#   family in {global CLS/full_redesign, patch location_kmeans}
#   mode in {frozen+timm eval transforms, finetunable legacy transforms}
#
# Usage:
#   .\run_gelu_dropout_sweep.ps1
#   .\run_gelu_dropout_sweep.ps1 -SkipCompleted
#   .\run_gelu_dropout_sweep.ps1 -StopOnFailure

param(
    [switch]$SkipCompleted,
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MPLBACKEND = "Agg"
$env:HF_HUB_OFFLINE = "1"
$env:LOKY_MAX_CPU_COUNT = "8"

$root = "d:\Documents\FMI\Disertatie\medical-ad"
$python = Join-Path $root "venv\Scripts\python.exe"
$main = Join-Path $root "project\main.py"
$runsRoot = Join-Path $root "runs"
$logDir = Join-Path $runsRoot "gelu_dropout_sweep_logs"
$timingPath = Join-Path $runsRoot "gelu_dropout_sweep_timings.json"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$experiments = @(
    @{ config = "project\configs\gelu_dropout_global_k1_frozen_timm.yaml";            name = "gelu_dropout_global_k1_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_global_k1_finetune.yaml";               name = "gelu_dropout_global_k1_finetune" },
    @{ config = "project\configs\gelu_dropout_patch_location_k1_frozen_timm.yaml";    name = "gelu_dropout_patch_location_k1_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_patch_location_k1_finetune.yaml";       name = "gelu_dropout_patch_location_k1_finetune" },

    @{ config = "project\configs\gelu_dropout_global_k64_frozen_timm.yaml";           name = "gelu_dropout_global_k64_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_global_k64_finetune.yaml";              name = "gelu_dropout_global_k64_finetune" },
    @{ config = "project\configs\gelu_dropout_patch_location_k64_frozen_timm.yaml";   name = "gelu_dropout_patch_location_k64_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_patch_location_k64_finetune.yaml";      name = "gelu_dropout_patch_location_k64_finetune" },

    @{ config = "project\configs\gelu_dropout_global_k1024_frozen_timm.yaml";         name = "gelu_dropout_global_k1024_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_global_k1024_finetune.yaml";            name = "gelu_dropout_global_k1024_finetune" },
    @{ config = "project\configs\gelu_dropout_patch_location_k1024_frozen_timm.yaml"; name = "gelu_dropout_patch_location_k1024_frozen_timm" },
    @{ config = "project\configs\gelu_dropout_patch_location_k1024_finetune.yaml";    name = "gelu_dropout_patch_location_k1024_finetune" }
)

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

function Get-ExperimentDir {
    param(
        [string]$ExperimentName
    )

    $nested = Join-Path (Join-Path $runsRoot $ExperimentName) $ExperimentName
    if (Test-Path $nested) {
        return $nested
    }

    return (Join-Path $runsRoot $ExperimentName)
}

function Test-ExperimentCompleted {
    param(
        [string]$ExperimentName
    )

    $experimentDir = Get-ExperimentDir -ExperimentName $ExperimentName
    $metricsPath = Join-Path $experimentDir "evaluation\evaluation_metrics.json"
    $stage2Model = Join-Path $experimentDir "final_stage2_model.pth"
    $stage1Model = Join-Path $experimentDir "final_model.pth"
    return ((Test-Path $metricsPath) -and ((Test-Path $stage2Model) -or (Test-Path $stage1Model)))
}

Set-Location $root

foreach ($exp in $experiments) {
    $configPath = Join-Path $root $exp.config
    $logPath = Join-Path $logDir "$($exp.name).log"
    $tmpSuffix = [System.Guid]::NewGuid().ToString("N")
    $stdoutPath = Join-Path $logDir "$($exp.name).$tmpSuffix.stdout.tmp"
    $stderrPath = Join-Path $logDir "$($exp.name).$tmpSuffix.stderr.tmp"

    if ($SkipCompleted -and (Test-ExperimentCompleted -ExperimentName $exp.name)) {
        $skippedAt = Get-Date
        $entry = [ordered]@{
            name = $exp.name
            config = $configPath
            log_path = $logPath
            started_at = $skippedAt.ToString("o")
            finished_at = $skippedAt.ToString("o")
            elapsed_minutes = 0.0
            total_elapsed_minutes = [math]::Round(($skippedAt - $runStartedAt).TotalMinutes, 2)
            exit_code = 0
            status = "skipped_completed"
        }
        [void]$timings.Add($entry)
        Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "running"
        Write-Host "SKIP: $($exp.name) already completed" -ForegroundColor Yellow
        continue
    }

    $expStartedAt = Get-Date

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Starting: $($exp.name)" -ForegroundColor Cyan
    Write-Host "  Config: $configPath" -ForegroundColor DarkCyan
    Write-Host "  Log: $logPath" -ForegroundColor DarkCyan
    Write-Host "  Started at: $($expStartedAt.ToString('u'))" -ForegroundColor DarkCyan
    Write-Host "================================================================" -ForegroundColor Cyan

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @($main, "--config", $configPath, "--exp-name", $exp.name) `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    $exitCode = $process.ExitCode

    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($logPath, "STDOUT`r`n======`r`n", $encoding)
    if (Test-Path $stdoutPath) {
        [System.IO.File]::AppendAllText($logPath, [System.IO.File]::ReadAllText($stdoutPath), $encoding)
    }
    [System.IO.File]::AppendAllText($logPath, "`r`nSTDERR`r`n======`r`n", $encoding)
    if (Test-Path $stderrPath) {
        [System.IO.File]::AppendAllText($logPath, [System.IO.File]::ReadAllText($stderrPath), $encoding)
    }
    Remove-Item -Path $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

    $expFinishedAt = Get-Date
    $elapsedMinutes = [math]::Round(($expFinishedAt - $expStartedAt).TotalMinutes, 2)
    $totalMinutes = [math]::Round(($expFinishedAt - $runStartedAt).TotalMinutes, 2)

    $entry = [ordered]@{
        name = $exp.name
        config = $configPath
        log_path = $logPath
        started_at = $expStartedAt.ToString("o")
        finished_at = $expFinishedAt.ToString("o")
        elapsed_minutes = $elapsedMinutes
        total_elapsed_minutes = $totalMinutes
        exit_code = $exitCode
        status = if ($exitCode -eq 0) { "completed" } else { "failed" }
    }
    [void]$timings.Add($entry)
    Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "running"

    if ($exitCode -ne 0) {
        $hadFailures = $true
        Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "running_with_failures"
        Write-Host "FAILED: $($exp.name) (exit $exitCode, elapsed $elapsedMinutes min)" -ForegroundColor Red
        if ($StopOnFailure) {
            Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status "failed"
            Write-Host "Timing log saved to: $timingPath" -ForegroundColor Yellow
            exit $exitCode
        }
        Write-Host "Continuing because -StopOnFailure was not set." -ForegroundColor Yellow
        continue
    }

    Write-Host "DONE: $($exp.name)" -ForegroundColor Green
    Write-Host "  Experiment elapsed: $elapsedMinutes min" -ForegroundColor Green
    Write-Host "  Total elapsed: $totalMinutes min" -ForegroundColor Green
}

$finalStatus = if ($hadFailures) { "completed_with_failures" } else { "completed" }
Save-TimingSummary -StartedAt $runStartedAt -Entries $timings -Path $timingPath -Status $finalStatus

$finishedAt = Get-Date
$totalRunMinutes = [math]::Round(($finishedAt - $runStartedAt).TotalMinutes, 2)

Write-Host ""
if ($hadFailures) {
    Write-Host "GELU/dropout sweep attempted all experiments, with failures." -ForegroundColor Yellow
} else {
    Write-Host "All GELU/dropout sweep experiments complete." -ForegroundColor Green
}
Write-Host "Total runtime: $totalRunMinutes min" -ForegroundColor Green
Write-Host "Timing log: $timingPath" -ForegroundColor Green
Write-Host "Logs: $logDir" -ForegroundColor Green

if ($hadFailures) {
    exit 1
}
