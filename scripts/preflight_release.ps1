# Local PyPI release preflight (Windows). Does not upload.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "== matching-pmh release preflight ==" -ForegroundColor Cyan
pip install -q build twine
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
python -m build
twine check dist/*
Write-Host "OK: dist/ ready for: twine upload dist/*" -ForegroundColor Green
