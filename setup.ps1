Write-Host "========================================"
Write-Host "  Adversarial Detection System Setup"
Write-Host "  60% Milestone Implementation"
Write-Host "========================================"
Write-Host ""

python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python was not found."
    exit 1
}

Write-Host "[INFO] Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed."
    exit 1
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Run demo: python demo_60.py"
Write-Host "  2. Start web interface: python web/app.py"
Write-Host "  3. Open browser: http://localhost:5000"
