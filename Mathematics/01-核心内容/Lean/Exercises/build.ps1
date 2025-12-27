param(
  [switch]$Clean
)

Write-Host "[Lean Exercises] Build start" -ForegroundColor Cyan

if ($Clean) {
  if (Test-Path .lake) { Remove-Item -Recurse -Force .lake }
  if (Test-Path build) { Remove-Item -Recurse -Force build }
}

if (-not (Get-Command lean -ErrorAction SilentlyContinue)) {
  Write-Warning "lean 未安装或未在 PATH。请安装 elan 并重启终端：https://leanprover-community.github.io/get_started.html"
}

if (-not (Get-Command lake -ErrorAction SilentlyContinue)) {
  Write-Warning "lake 未安装或未在 PATH。请参考 README 的 Windows 环境准备步骤。"
  exit 1
}

lake build

if ($LASTEXITCODE -ne 0) {
  Write-Error "lake build 失败。可尝试：lake clean；检查 lakefile.lean 的 mathlib 固定版本。"
  exit $LASTEXITCODE
}

Write-Host "[Lean Exercises] Build success" -ForegroundColor Green

