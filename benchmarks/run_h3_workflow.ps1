param(
    [string]$Workflow = "C:\Users\Administrator\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Minimax_H3_turbo_workflow_fixed.json",
    [int]$Runs = 2,
    [int]$Port = 8189,
    [string]$LogPrefix = "comfy-auto",
    [switch]$KeepAlive,
    [switch]$SkipRun,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\Administrator\ComfyUI_windows_portable"
$python = Join-Path $root "python_embeded\python.exe"
$comfy = Join-Path $root "python_embeded\Scripts\comfy.exe"
$mainPy = Join-Path $root "ComfyUI\main.py"

if (-not (Test-Path $Workflow)) { throw "workflow not found: $Workflow" }
if (-not (Test-Path $comfy)) { throw "comfy CLI not found: $comfy" }

# Mirror run_intel_gpu.bat
$env:OMNI_ATTN_BACKEND = "esimd"
$env:OMNIXPU_ENABLE = "1"
$env:OMNIXPU_ATTENTION = "1"
$env:COMFYUI_GGUF_BACKEND = "xpu"
$env:OMNIXPU_DEBUG_VERBOSE = if ($Verbose) { "1" } else { "0" }

$logFile = Join-Path $root "$LogPrefix.log"
$outFile = Join-Path $root "$LogPrefix.stdout.txt"
$errFile = Join-Path $root "$LogPrefix.stderr.txt"
Remove-Item $logFile, $outFile, $errFile -ErrorAction SilentlyContinue

$args = @("-s", "ComfyUI\main.py", "--windows-standalone-build",
          "--enable-manager", "--enable-dynamic-vram",
          "--verbose", "INFO", ".\$LogPrefix.log", "--port", "$Port")
Write-Host "Starting ComfyUI on port $Port ..."
$proc = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru

try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($proc.HasExited) { throw "ComfyUI exited early (code $($proc.ExitCode))" }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw "ComfyUI did not become ready on port $Port" }
    Write-Host "ComfyUI ready."

    if ($SkipRun) { Write-Host "Skipping workflow runs (validate only)."; exit 0 }

    for ($run = 1; $run -le $Runs; $run++) {
        Write-Host "=== run $run/$Runs ==="
        & $comfy run --workflow $Workflow --wait --port $Port --no-notify
        if ($LASTEXITCODE -ne 0) { throw "comfy run failed with exit code $LASTEXITCODE" }
        $m = Select-String -Path $logFile -Pattern "Prompt executed in ([\d\.]+) seconds" | Select-Object -Last 1
        if ($m) { Write-Host ("run {0} completed: {1} seconds" -f $run, $m.Matches[0].Groups[1].Value) }
    }
} finally {
    if (-not $KeepAlive -and -not $proc.HasExited) {
        Write-Host "Stopping ComfyUI (pid $($proc.Id))"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}
