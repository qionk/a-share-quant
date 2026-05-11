@echo off
REM ============================================================
REM  A-Share Quant 本地训练一键启动脚本 (Windows CMD)
REM  使用方法: 双击运行或在 CMD 中执行 setup_and_train.bat
REM ============================================================

echo.
echo ================================================================
echo   A-Share Quant 本地训练环境配置 & 启动
echo   %date% %time%
echo ================================================================
echo.

REM ── 1. 设置 MySQL 环境变量 ──────────────────────────────
echo [步骤 1/4] 配置 MySQL 数据库连接...
set MYSQL_HOST=mysql3.sqlpub.com
set MYSQL_PORT=3308
set MYSQL_USER=root_quant
set MYSQL_PASSWORD=BLnVlQ8qASfhA9xZ
set MYSQL_DATABASE=a_share_quant
echo   MYSQL_HOST      = %MYSQL_HOST%
echo   MYSQL_PORT      = %MYSQL_PORT%
echo   MYSQL_USER      = %MYSQL_USER%
echo   MYSQL_DATABASE  = %MYSQL_DATABASE%
echo   [OK] 环境变量已设置
echo.

REM ── 2. 检查 Python ────────────────────────────────────────
echo [步骤 2/4] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [错误] 未找到 Python! 请先安装 Python 3.11+
    echo   下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo   [OK] Python 可用
echo.

REM ── 3. 安装/更新依赖 ─────────────────────────────────────
echo [步骤 3/4] 安装 Python 依赖...
echo   正在执行: pip install -r requirements.txt
echo   (如果已安装会自动跳过)
echo.
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo   [警告] pip install 有警告，但可能不影响运行
    echo   如果后续训练失败，请手动执行:
    echo     pip install -r requirements.txt
)
echo   [OK] 依赖安装完成
echo.

REM ── 4. 启动训练 ──────────────────────────────────────────
echo [步骤 4/4] 启动批量训练...
echo ================================================================
echo.
python scripts\train_all_stocks.py
echo.
echo ================================================================
echo   训练进程结束
echo ================================================================
pause