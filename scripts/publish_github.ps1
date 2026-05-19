# Full publish: GitHub login -> create vishalstark512/matching-pmh -> push main + tag
# Run in PowerShell:  cd C:\Users\Eigenaar\Desktop\matching-pmh
#                     .\scripts\publish_github.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) {
    $c = Get-Command gh -ErrorAction SilentlyContinue
    if ($c) { $gh = $c.Source } else { throw "Install GitHub CLI: winget install GitHub.cli" }
}

Write-Host "`n=== Step 1: GitHub login ===" -ForegroundColor Cyan
& $gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Browser will open. Approve login for account vishalstark512." -ForegroundColor Yellow
    & $gh auth login -h github.com -p https -w
}
& $gh auth status
if ($LASTEXITCODE -ne 0) { throw "GitHub login failed. Run: gh auth login" }

Write-Host "`n=== Step 2: Prepare git ===" -ForegroundColor Cyan
git branch -M main
git remote remove origin 2>$null
git remote add origin https://github.com/vishalstark512/matching-pmh.git
git remote -v

Write-Host "`n=== Step 3: Create repo on GitHub + push main ===" -ForegroundColor Cyan
$create = & $gh repo create vishalstark512/matching-pmh --public --source=. --remote=origin --push 2>&1
$create | Write-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "Create skipped or repo exists; pushing with git..." -ForegroundColor Yellow
    git push -u origin main
}

Write-Host "`n=== Step 4: Push tag v0.6.0 ===" -ForegroundColor Cyan
git push origin v0.6.0

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "https://github.com/vishalstark512/matching-pmh"
& $gh repo view vishalstark512/matching-pmh 2>&1 | Select-Object -First 8
