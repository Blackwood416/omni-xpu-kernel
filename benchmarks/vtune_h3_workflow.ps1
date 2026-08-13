param(
    [string]$Workflow = "C:\Users\Administrator\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Minimax_H3_turbo_workflow_fixed.json",
    [int]$Port = 8189,
    [string]$ResultDir = "C:\Temp\vtune_h3wf",
    [int]$DurationSec = 360
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\Administrator\ComfyUI_windows_portable"
$python = Join-Path $root "python_embeded\python.exe"
$comfy = Join-Path $root "python_embeded\Scripts\comfy.exe"

$env:OMNI_ATTN_BACKEND = "esimd"
$env:OMNIXPU_ENABLE = "1"
$env:OMNIXPU_ATTENTION = "1"
$env:COMFYUI_GGUF_BACKEND = "xpu"
$env:OMNIXPU_DEBUG_VERBOSE = "0"

$logFile = Join-Path $root "comfy-vtune.log"
$outFile = Join-Path $root "comfy-vtune.stdout.txt"
$errFile = Join-Path $root "comfy-vtune.stderr.txt"
Remove-Item $logFile, $outFile, $errFile -ErrorAction SilentlyContinue
if (Test-Path $ResultDir) { Remove-Item $ResultDir -Recurse -Force -ErrorAction SilentlyContinue }

# Seed-randomized workflow copy so the single submitted run is not cached.
$json = Get-Content $Workflow -Raw | ConvertFrom-Json
$noise = $json.nodes | Where-Object { $_.type -eq "RandomNoise" } | Select-Object -First 1
if ($noise -and $noise.widgets_values.Count -gt 0) {
    $noise.widgets_values[0] = Get-Random -Minimum 1000000000000 -Maximum 9999999999999
}
$runWf = Join-Path $env:TEMP "h3_workflow_vtune.json"
[System.IO.File]::WriteAllText($runWf, ($json | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))

$mainArgs = @("-s", "ComfyUI\main.py", "--windows-standalone-build",
              "--enable-manager", "--enable-dynamic-vram",
              "--verbose", "INFO", ".\comfy-vtune.log", "--port", "$Port")
$vtuneArgs = @("-collect", "gpu-hotspots", "-duration", "$DurationSec",
               "-result-dir", $ResultDir, "--", $python) + $mainArgs

Write-Host "Starting VTune collection (gpu-hotspots, up to ${DurationSec}s) ..."
$vtune = Start-Process -FilePath "vtune" -ArgumentList $vtuneArgs `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($vtune.HasExited) { throw "vtune exited early (code $($vtune.ExitCode))" }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw "ComfyUI did not become ready on port $Port" }
    Write-Host "ComfyUI ready; submitting workflow under VTune ..."
    & $comfy run --workflow $runWf --wait --port $Port --no-notify
    if ($LASTEXITCODE -ne 0) { throw "comfy run failed (exit $LASTEXITCODE)" }
} finally {
    Write-Host "Waiting for VTune collection to finish ..."
    if (-not $vtune.HasExited) { $vtune.WaitForExit(600000) | Out-Null }
}

Write-Host "=== summary ==="
vtune -report summary -r $ResultDir
