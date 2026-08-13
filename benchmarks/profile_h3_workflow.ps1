param(
    [string]$Workflow = "C:\Users\Administrator\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Minimax_H3_turbo_workflow_fixed.json",
    [int]$Port = 8189,
    [int]$SampleDelaySec = 45,
    [int]$SampleDurationSec = 120
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\Administrator\ComfyUI_windows_portable"
$python = Join-Path $root "python_embeded\python.exe"
$comfy = Join-Path $root "python_embeded\Scripts\comfy.exe"
$pyspy = Join-Path $root "python_embeded\Scripts\py-spy.exe"

$env:OMNI_ATTN_BACKEND = "esimd"
$env:OMNIXPU_ENABLE = "1"
$env:OMNIXPU_ATTENTION = "1"
$env:COMFYUI_GGUF_BACKEND = "xpu"
$env:OMNIXPU_DEBUG_VERBOSE = "0"

$logFile = Join-Path $root "comfy-profile.log"
$outFile = Join-Path $root "comfy-profile.stdout.txt"
$errFile = Join-Path $root "comfy-profile.stderr.txt"
Remove-Item $logFile, $outFile, $errFile -ErrorAction SilentlyContinue

# Seed-randomized workflow copy.
$json = Get-Content $Workflow -Raw | ConvertFrom-Json
$noise = $json.nodes | Where-Object { $_.type -eq "RandomNoise" } | Select-Object -First 1
if ($noise -and $noise.widgets_values.Count -gt 0) {
    $noise.widgets_values[0] = Get-Random -Minimum 1000000000000 -Maximum 9999999999999
}
$runWf = Join-Path $env:TEMP "h3_workflow_profile.json"
[System.IO.File]::WriteAllText($runWf, ($json | ConvertTo-Json -Depth 100), (New-Object System.Text.UTF8Encoding($false)))

$mainArgs = @("-s", "ComfyUI\main.py", "--windows-standalone-build",
              "--enable-manager", "--enable-dynamic-vram",
              "--verbose", "INFO", ".\comfy-profile.log", "--port", "$Port")
$proc = Start-Process -FilePath $python -ArgumentList $mainArgs -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($proc.HasExited) { throw "ComfyUI exited early" }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw "ComfyUI not ready" }

    Write-Host "Submitting workflow ..."
    $run = Start-Process -FilePath $comfy -ArgumentList @("run", "--workflow", $runWf, "--wait", "--port", "$Port", "--no-notify") `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root "comfy-profile.run.txt") `
        -RedirectStandardError (Join-Path $root "comfy-profile.runerr.txt") -PassThru

    Start-Sleep -Seconds $SampleDelaySec
    Write-Host "Sampling pid $($proc.Id) for ${SampleDurationSec}s ..."
    $svg = Join-Path $root "comfy-profile.svg"
    & $pyspy record --pid $proc.Id --duration $SampleDurationSec --output $svg
    $run.WaitForExit(1200000) | Out-Null
    Write-Host "workflow done: $($run.ExitCode)"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
