# ============================================================
#  A-Share Quant 本地训练一键启动脚本 (Git Bash)
#  使用方法: bash scripts/setup_and_train.sh
# ============================================================

echo ""
echo "============================================================"
echo "  A-Share Quant 本地训练环境配置 & 启动"
echo "  $(date)"
echo "============================================================"
echo ""

# ── 1. 设置 MySQL 环境变量 ──────────────────────────────
echo "[步骤 1/4] 配置 MySQL 数据库连接..."
export MYSQL_HOST="mysql3.sqlpub.com"
export MYSQL_PORT="3308"
export MYSQL_USER="root_quant"
export MYSQL_PASSWORD="BLnVlQ8qASfhA9xZ"
export MYSQL_DATABASE="a_share_quant"
echo "  MYSQL_HOST      = $MYSQL_HOST"
echo "  MYSQL_PORT      = $MYSQL_PORT"
echo "  MYSQL_USER      = $MYSQL_USER"
echo "  MYSQL_DATABASE  = $MYSQL_DATABASE"
echo "  [OK] 环境变量已设置"
echo ""

# ── 2. 检查 Python ────────────────────────────────────────
echo "[步骤 2/4] 检查 Python 环境..."
if ! command -v python &> /dev/null; then
    echo "  [错误] 未找到 Python! 请先安装 Python 3.11+"
    exit 1
fi
python --version
echo "  [OK] Python 可用"
echo ""

# ── 3. 安装/更新依赖 ─────────────────────────────────────
echo "[步骤 3/4] 安装 Python 依赖..."
pip install -r requirements.txt --quiet
echo "  [OK] 依赖安装完成"
echo ""

# ── 4. 启动训练 ──────────────────────────────────────────
echo "[步骤 4/4] 启动批量训练..."
echo "============================================================"
echo ""
python scripts/train_all_stocks.py
echo ""
echo "============================================================"
echo "  训练进程结束"
echo "============================================================"