# Upload matching-pmh to PyPI. Run from repo root in YOUR terminal (token stays local).
#
#   cd C:\Users\Eigenaar\Desktop\matching-pmh
#   .\scripts\upload_pypi.ps1
#
# Or set env vars first (no prompt):
#   $env:TWINE_USERNAME = "__token__"
#   $env:TWINE_PASSWORD = "pypi-..."
#   .\scripts\upload_pypi.ps1 -SkipPrompt

param(
    [switch]$TestPyPI,
    [switch]$SkipBuild,
    [switch]$SkipPrompt
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not $env:TWINE_PASSWORD -and -not $SkipPrompt) {
    Write-Host ""
    Write-Host "PyPI API token: https://pypi.org/manage/account/ -> API tokens -> Add API token" -ForegroundColor Cyan
    Write-Host "  Scope: entire account (first upload) or project matching-pmh" -ForegroundColor DarkGray
    Write-Host "  Username is always: __token__" -ForegroundColor DarkGray
    Write-Host ""
    $secure = Read-Host "Paste token (pypi-...)" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $env:TWINE_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    $env:TWINE_USERNAME = "__token__"
}

if (-not $env:TWINE_PASSWORD) {
    Write-Host "No token. Set TWINE_PASSWORD or run without -SkipPrompt." -ForegroundColor Red
    exit 1
}
if (-not $env:TWINE_USERNAME) {
    $env:TWINE_USERNAME = "__token__"
}

pip install -q build twine

if (-not $SkipBuild) {
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    python -m build
}
twine check dist\*

$target = if ($TestPyPI) { "TestPyPI" } else { "PyPI (production)" }
Write-Host ""
$ver = python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
Write-Host "Uploading matching-pmh $ver to $target ..." -ForegroundColor Cyan
if ($TestPyPI) {
    twine upload --repository testpypi dist\*
} else {
    twine upload dist\*
}

Write-Host ""
Write-Host "Published successfully." -ForegroundColor Green
if ($TestPyPI) {
    Write-Host "  pip install -i https://test.pypi.org/simple/ matching-pmh==$ver"
} else {
    Write-Host "  pip install matching-pmh==$ver"
    Write-Host '  pmh-train list-methods'
    Write-Host '  https://pypi.org/project/matching-pmh/'
}
