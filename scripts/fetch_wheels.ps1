# Resumable wheel downloader.
#
# This machine gets ~100 KB/s to PyPI and pip has no resume support, so a
# 50MB wheel that drops mid-transfer restarts from zero every time. curl's
# -C - resumes from wherever the previous attempt died, which makes large
# wheels actually attainable on a link this slow.
#
# Usage:  powershell -File scripts\fetch_wheels.ps1

$ErrorActionPreference = "Stop"
$dest = "D:\ecommerce-ml-platform\.wheels"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# package, version, and a regex matching the wheel we need for cp313 win_amd64
$wanted = @(
    @{ name = "polars";            version = "1.43.2"; pattern = "py3-none-any" },
    @{ name = "polars-runtime-32"; version = "1.43.2"; pattern = "abi3-win_amd64" },
    @{ name = "duckdb";            version = "1.5.5";  pattern = "cp313-cp313-win_amd64" }
)

foreach ($w in $wanted) {
    $api = "https://pypi.org/pypi/$($w.name)/$($w.version)/json"
    Write-Host "resolving $($w.name)==$($w.version) ..." -ForegroundColor Cyan

    $json = curl.exe -sS --max-time 90 --retry 5 --retry-delay 3 $api | ConvertFrom-Json
    $file = $json.urls | Where-Object { $_.filename -match $w.pattern -and $_.packagetype -eq "bdist_wheel" } | Select-Object -First 1

    if (-not $file) {
        Write-Host "  NO MATCH for pattern '$($w.pattern)'. Available:" -ForegroundColor Red
        $json.urls | ForEach-Object { Write-Host "    $($_.filename)" }
        continue
    }

    $out = Join-Path $dest $file.filename
    $sizeMB = [math]::Round($file.size / 1MB, 1)

    if ((Test-Path $out) -and ((Get-Item $out).Length -eq $file.size)) {
        Write-Host "  already complete: $($file.filename) ($sizeMB MB)" -ForegroundColor Green
        continue
    }

    Write-Host "  downloading $($file.filename) ($sizeMB MB) with resume ..." -ForegroundColor Yellow
    # -C -    resume from wherever we left off
    # --retry re-establish on transient failures, with backoff
    curl.exe -L -C - --retry 30 --retry-delay 5 --retry-all-errors `
             --connect-timeout 30 --max-time 3600 `
             -o $out $file.url

    if (Test-Path $out) {
        $have = (Get-Item $out).Length
        if ($have -eq $file.size) {
            Write-Host "  OK $($file.filename)" -ForegroundColor Green
        } else {
            Write-Host ("  INCOMPLETE {0}: {1:N0} of {2:N0} bytes - rerun to resume" -f $file.filename, $have, $file.size) -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "wheels in $dest :" -ForegroundColor Cyan
Get-ChildItem $dest -Filter *.whl | ForEach-Object { "  {0,-55} {1,7:N1} MB" -f $_.Name, ($_.Length/1MB) }
