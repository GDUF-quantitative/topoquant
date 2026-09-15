[CmdletBinding()]
param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$OutputRoot = "dist",
    [switch]$WithoutData,
    [switch]$SkipSelfTest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$PythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
& $PythonPath -c "import PyInstaller"
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is missing. Run: .venv\Scripts\python -m pip install -e ".[portable]"'
}

$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
$BuildPath = Join-Path $ProjectRoot "build\pyinstaller"

& $PythonPath -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name "TopoQuant" `
    --contents-directory "runtime" `
    --paths (Join-Path $ProjectRoot "src") `
    --collect-all "topp" `
    --collect-all "ripser" `
    --copy-metadata "topp" `
    --copy-metadata "ripser" `
    --copy-metadata "numpy" `
    --copy-metadata "pandas" `
    --copy-metadata "rich" `
    --distpath $OutputPath `
    --workpath $BuildPath `
    --specpath $BuildPath `
    (Join-Path $ProjectRoot "run.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$PackagePath = Join-Path $OutputPath "TopoQuant"
Copy-Item -LiteralPath (Join-Path $ProjectRoot "config.example.json") -Destination $PackagePath -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "portable\start_topoquant.cmd") -Destination $PackagePath -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "portable\check_runtime.cmd") -Destination $PackagePath -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\PORTABLE.md") -Destination (Join-Path $PackagePath "README.md") -Force

$StockTarget = Join-Path $PackagePath "data\stock"
New-Item -ItemType Directory -Path $StockTarget -Force | Out-Null
if (-not $WithoutData) {
    $StockSource = Join-Path $ProjectRoot "data\stock"
    $StockFiles = @(Get-ChildItem -LiteralPath $StockSource -Filter "*.csv" -File)
    if ($StockFiles.Count -eq 0) {
        throw "No CSV files were found in data\stock. Use -WithoutData to build an empty package."
    }
    Copy-Item -Path (Join-Path $StockSource "*.csv") -Destination $StockTarget -Force
    $PackagedStockFiles = @(Get-ChildItem -LiteralPath $StockTarget -Filter "*.csv" -File)
    $SourceBytes = ($StockFiles | Measure-Object -Property Length -Sum).Sum
    $PackagedBytes = ($PackagedStockFiles | Measure-Object -Property Length -Sum).Sum
    if (
        $StockFiles.Count -ne $PackagedStockFiles.Count -or
        $SourceBytes -ne $PackagedBytes
    ) {
        throw "Portable data copy verification failed"
    }
}

if (-not $SkipSelfTest) {
    & (Join-Path $PackagePath "TopoQuant.exe") --portable-self-test
    if ($LASTEXITCODE -ne 0) { throw "Portable executable self-test failed" }
    & (Join-Path $PackagePath "TopoQuant.exe") --portable-pipeline-self-test
    if ($LASTEXITCODE -ne 0) { throw "Portable multiprocessing pipeline self-test failed" }
}

$RuntimeSize = (Get-ChildItem -LiteralPath $PackagePath -Recurse -File | Measure-Object -Property Length -Sum).Sum
$RuntimeSizeMiB = [math]::Round($RuntimeSize / 1MB, 1)
Write-Output "Portable package ready: $PackagePath"
Write-Output "Files: $RuntimeSizeMiB MiB; data included: $(-not $WithoutData)"
