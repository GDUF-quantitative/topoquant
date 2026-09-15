[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$Config = "",
    [string]$Wheelhouse = "",
    [switch]$CheckOnly,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$SelectedCommand = $null
$SelectedPrefix = @()

if ($PythonExe) {
    if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }
    $SelectedCommand = $PythonExe
} else {
    $Candidates = @(
        @{ Command = "py"; Prefix = @("-3.13") },
        @{ Command = "py"; Prefix = @("-3.12") },
        @{ Command = "py"; Prefix = @("-3.11") },
        @{ Command = "py"; Prefix = @("-3.10") },
        @{ Command = "python"; Prefix = @() },
        @{ Command = "python3"; Prefix = @() }
    )
    foreach ($Candidate in $Candidates) {
        $CandidateCommand = $Candidate.Command
        $CandidatePrefix = $Candidate.Prefix
        if (-not (Get-Command $CandidateCommand -ErrorAction SilentlyContinue)) {
            continue
        }
        & $CandidateCommand @CandidatePrefix -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $SelectedCommand = $CandidateCommand
            $SelectedPrefix = $CandidatePrefix
            break
        }
    }
}

if (-not $SelectedCommand) {
    throw "Python >= 3.10 was not found. Install 64-bit Python from https://www.python.org/downloads/windows/"
}

& $SelectedCommand @SelectedPrefix -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python must be version 3.10 or newer" }

& $SelectedCommand @SelectedPrefix -c "import sys; print('Using Python', sys.version, 'at', sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "Python check failed" }
if ($CheckOnly) {
    Write-Output "Python prerequisite check passed"
    exit 0
}

& $SelectedCommand @SelectedPrefix -m venv ".venv"
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$InstallTarget = "."
if ($Dev) { $InstallTarget = ".[dev]" }
if ($Wheelhouse) {
    $ResolvedWheelhouse = (Resolve-Path -LiteralPath $Wheelhouse).Path
    & $VenvPython -m pip install --no-index --find-links $ResolvedWheelhouse pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Offline pip bootstrap failed" }
    & $VenvPython -m pip install --no-index --find-links $ResolvedWheelhouse -e $InstallTarget
} else {
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
    & $VenvPython -m pip install -e $InstallTarget
}
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed" }

& $VenvPython -m topoquant --help
if ($LASTEXITCODE -ne 0) { throw "TopoQuant sanity check failed" }

if ($Config) {
    & $VenvPython -m topoquant --config $Config preflight
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed; review the table above" }
}

Write-Output "Installation complete. Python: $VenvPython"
