param(
    [Parameter(Mandatory = $true)]
    [string]$LauncherPath,

    [string]$WorkspaceRoot = "",

    [int]$PollTimeoutSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    Write-Host "[windows-launcher-smoke] $Message"
}

function Resolve-ExistingPath {
    param(
        [string]$PathValue,
        [string]$Label
    )
    try {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    catch {
        throw "$Label not found: $PathValue"
    }
}

function Resolve-PythonPath {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source)) {
        return $cmd.Source
    }
    throw "python was not found in PATH. The Windows launcher smoke test requires a runnable Python interpreter."
}

function Invoke-LoggedCommand {
    param(
        [string]$ExePath,
        [string[]]$Arguments,
        [string]$StepName,
        [string]$LogPath
    )

    Write-Info "$StepName"
    Write-Info ("command: {0} {1}" -f $ExePath, ($Arguments -join " "))
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $ExePath @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    if ($exitCode -ne 0) {
        throw ("{0} failed with exit={1}. See log: {2}" -f $StepName, $exitCode, $LogPath)
    }
    Write-Info ("{0} completed in {1:N1}s" -f $StepName, $sw.Elapsed.TotalSeconds)
}

function Wait-Until {
    param(
        [scriptblock]$Condition,
        [string]$Description,
        [int]$TimeoutSeconds,
        [int]$IntervalMilliseconds = 500
    )

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if (& $Condition) {
            Write-Info ("{0} satisfied after {1:N1}s" -f $Description, $sw.Elapsed.TotalSeconds)
            return
        }
        Start-Sleep -Milliseconds $IntervalMilliseconds
    }
    throw ("Timed out waiting for {0} after {1}s" -f $Description, $TimeoutSeconds)
}

$repoRoot = Resolve-ExistingPath -PathValue (Join-Path $PSScriptRoot "..\..") -Label "repo root"
$launcherSource = Resolve-ExistingPath -PathValue $LauncherPath -Label "launcher binary"
$pythonPath = Resolve-PythonPath

if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("jx-launcher-smoke-{0}-{1}" -f $PID, (Get-Date -Format "yyyyMMddHHmmss"))
}

$workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
$installDir = Join-Path $workspace "install"
$runtimeHome = Join-Path $workspace "runtime"
$logDir = Join-Path $workspace "logs"
$installedLauncher = Join-Path $installDir "jx.exe"
$stagedLauncher = Join-Path $installDir "jx.new.exe"
$backupLauncher = Join-Path $installDir "jx.previous.exe"
$replaceHelper = Join-Path $installDir "jx_replace_launcher.cmd"

if (Test-Path -LiteralPath $workspace) {
    Remove-Item -LiteralPath $workspace -Recurse -Force
}
New-Item -ItemType Directory -Path $installDir -Force | Out-Null
New-Item -ItemType Directory -Path $runtimeHome -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Copy-Item -LiteralPath $launcherSource -Destination $installedLauncher -Force

$env:JX_HOME = $runtimeHome
$env:JX_PYTHON = $pythonPath
$env:PYTHONUTF8 = "1"

Write-Info "repo root: $repoRoot"
Write-Info "workspace: $workspace"
Write-Info "install dir: $installDir"
Write-Info "runtime home: $runtimeHome"
Write-Info "python: $pythonPath"

Invoke-LoggedCommand `
    -ExePath $installedLauncher `
    -Arguments @("-v") `
    -StepName "Preflight launcher version probe" `
    -LogPath (Join-Path $logDir "01-preflight-version.log")

Invoke-LoggedCommand `
    -ExePath $installedLauncher `
    -Arguments @("-upgrade", "latest", "-verbose") `
    -StepName "Launcher upgrade smoke test (jx -upgrade latest)" `
    -LogPath (Join-Path $logDir "02-upgrade-latest.log")

Wait-Until `
    -Condition { (Test-Path -LiteralPath $installedLauncher) -and -not (Test-Path -LiteralPath $stagedLauncher) } `
    -Description "Windows staged launcher replacement" `
    -TimeoutSeconds $PollTimeoutSeconds

Wait-Until `
    -Condition { -not (Test-Path -LiteralPath $replaceHelper) } `
    -Description "launcher replacement helper cleanup" `
    -TimeoutSeconds $PollTimeoutSeconds

Invoke-LoggedCommand `
    -ExePath $installedLauncher `
    -Arguments @("-v") `
    -StepName "Post-upgrade launcher version probe" `
    -LogPath (Join-Path $logDir "03-post-upgrade-version.log")

Invoke-LoggedCommand `
    -ExePath $installedLauncher `
    -Arguments @("-uninstall", "-yes") `
    -StepName "Launcher uninstall smoke test (jx -uninstall -yes)" `
    -LogPath (Join-Path $logDir "04-uninstall.log")

Wait-Until `
    -Condition { -not (Test-Path -LiteralPath $installedLauncher) } `
    -Description "launcher self-delete after uninstall" `
    -TimeoutSeconds $PollTimeoutSeconds

Wait-Until `
    -Condition { -not (Test-Path -LiteralPath $runtimeHome) } `
    -Description "runtime home removal after uninstall" `
    -TimeoutSeconds $PollTimeoutSeconds

$unexpectedArtifacts = @()
foreach ($path in @($stagedLauncher, $backupLauncher, $replaceHelper, (Join-Path $installDir ".jx_home"), (Join-Path $installDir ".launcher_version"))) {
    if (Test-Path -LiteralPath $path) {
        $unexpectedArtifacts += $path
    }
}

if ($unexpectedArtifacts.Count -gt 0) {
    throw ("Unexpected launcher artifacts remain after uninstall: {0}" -f ($unexpectedArtifacts -join ", "))
}

if (Test-Path -LiteralPath $installDir) {
    $leftovers = @(Get-ChildItem -LiteralPath $installDir -Force)
    if ($leftovers.Count -gt 0) {
        $names = $leftovers | ForEach-Object { $_.FullName }
        throw ("Install directory is not empty after uninstall: {0}" -f ($names -join ", "))
    }
}

Write-Info "Windows launcher smoke test passed."
