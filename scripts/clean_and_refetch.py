#!/usr/bin/env python3
"""
清理数据库并重新拉取股票数据（只保留最近 500 个交易日）

步骤:
  1. 列出当前数据库中所有股票代码
  2. 清空 stock_daily_data / training_sessions / model_results 三张表
  3. 逐只股票从 AKShare 拉取全量数据，截取最后 500 天写入 MySQL

用法:
  set MYSQL_HOST=mysql3.sqlpub.com
  set MYSQL_PORT=3308
  set MYSQL_USER=root_quant
  set MYSQL_PASSWORD=BLnVlQ8qASfhA9xZ
  set MYSQL_DATABASE=a_share_quant

  python scripts/clean_and_refetch.py
"""
import sys, os, time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
os.environ.setdefault("MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306"))
os.environ.setdefault("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
os.environ.setdefault("MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", ""))
os.environ.setdefault("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))

import pymysql
import pandas as pd
import numpy as np

MAX_DAYS = 500  # 每只股票最多保留的交易天数

print(f"\n{'='*60}")
print(f"  数据库清理 & 重新拉取 (保留最近 {MAX_DAYS} 天)")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")


# ── 1. 连接数据库 ──────────────────────────────────────
def get_conn():
    host = os.environ.get("MYSQL_HOST", "")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "")

    if not host or not user or not database:
        print("[错误] MySQL 环境变量未完整设置!")
        sys.exit(1)

    return pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset="utf8mb4",
        connect_timeout=10, read_timeout=30, write_timeout=30,
        autocommit=True,
    )


conn = get_conn()
cur = conn.cursor()
print("\n[OK] MySQL 连接成功")


# ── 2. 列出当前所有股票 ────────────────────────────────
print("\n── 当前数据库中的股票 ──")
cur.execute("""
    SELECT stock_code, stock_name, COUNT(*) AS cnt,
           MIN(trade_date) AS start_date, MAX(trade_date) AS end_date
    FROM stock_daily_data
    GROUP BY stock_code, stock_name
    ORDER BY stock_code
""")
existing = [(r[0], r[1], r[2], r[3], r[4]) for r in cur.fetchall()]

if not existing:
    print("  数据库中没有股票数据，无需清理")
    print("  请先在 Streamlit 网页中添加股票，或手动指定股票代码列表")
    conn.close()
    sys.exit(0)

stock_codes = []
for code, name, cnt, start, end in existing:
    s_str = str(start) if hasattr(start, "strftime") else str(start)
    e_str = str(end) if hasattr(end, "strftime") else str(end)
    print(f"  {code}  {name:<12}  {cnt}天  ({s_str} ~ {e_str})")
    stock_codes.append(code)

print(f"\n共 {len(stock_codes)} 只股票")

# ── 2b. 查看训练数据 ────────────────────────────────────
cur.execute("SELECT COUNT(*) FROM training_sessions")
ts_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM model_results")
mr_count = cur.fetchone()[0]
print(f"训练记录: {ts_count} 条 sessions, {mr_count} 条 model_results")

# ── 3. 确认清库 ─────────────────────────────────────────
print(f"\n{'!'*60}")
print(f"  即将执行以下操作:")
print(f"  1. 清空 stock_daily_data 表 ({sum(c for _,_,c,_,_ in existing)} 条)")
print(f"  2. 清空 training_sessions 表 ({ts_count} 条)")
print(f"  3. 清空 model_results 表 ({mr_count} 条)")
print(f"  4. 重新拉取 {len(stock_codes)} 只股票，各保留最近 {MAX_DAYS} 天")
print(f"{'!'*60}")

confirm = input("\n确认执行? (输入 yes 继续): ").strip().lower()
if confirm != "yes":
    print("已取消")
    conn.close()
    sys.exit(0)


# ── 4. 清空表 ───────────────────────────────────────────
print("\n── 清理数据库 ──")

tables = ["model_results", "training_sessions", "stock_daily_data"]
for table in tables:
    t0 = time.time()
    cur.execute(f"DELETE FROM {table}")
    deleted = cur.rowcount
    print(f"  [OK] 清空 {table}: {deleted} 条 ({time.time()-t0:.1f}s)")

print("  清理完成")


# ── 5. 重新拉取数据 ────────────────────────────────────
print(f"\n── 重新拉取数据 (每只股票保留最近 {MAX_DAYS} 天) ──")

from src.predict.data_input import load_from_akshare, get_stock_name

INSERT_SQL = """
    INSERT INTO stock_daily_data
    (stock_code, stock_name, trade_date, open, high, low, close, volume, amount, pct_change, turnover)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""

def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None

success_list = []
fail_list = []

for idx, code in enumerate(stock_codes):
    name = ""
    try:
        print(f"\n[{idx+1}/{len(stock_codes)}] {code} ...")

        # 获取股票名称
        try:
            name = get_stock_name(code)
        except Exception:
            name = code

        # 拉取全量数据（AKShare 默认返回全部历史）
        df = load_from_akshare(code, "20200101", datetime.now().strftime("%Y%m%d"))
        if df is None or df.empty:
            fail_list.append((code, name, "无数据返回"))
            print(f"  [失败] 无数据")
            continue

        total_days = len(df)
        print(f"  拉取到 {total_days} 天数据")

        # 截取最后 MAX_DAYS 天
        if len(df) > MAX_DAYS:
            df = df.tail(MAX_DAYS)
            print(f"  截取最近 {MAX_DAYS} 天: "
                  f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  数据不足 {MAX_DAYS} 天，全部保留")

        # 写入 MySQL
        rows = []
        for idx_row, row in df.iterrows():
            trade_date = idx_row.date() if hasattr(idx_row, "date") else pd.Timestamp(idx_row).date()
            rows.append((
                code, name, trade_date,
                safe_float(row.get("open")),
                safe_float(row.get("high")),
                safe_float(row.get("low")),
                safe_float(row.get("close")),
                safe_float(row.get("volume")),
                safe_float(row.get("amount")),
                safe_float(row.get("pct_change")),
                safe_float(row.get("turnover")),
            ))

        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            cur.executemany(INSERT_SQL, batch)

        last_close = df["close"].iloc[-1]
        success_list.append((code, name, len(rows)))
        print(f"  [OK] {name} ({code})  写入 {len(rows)} 天  "
              f"最新价 {last_close:.2f}  "
              f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

        # 请求间隔，避免触发反爬
        time.sleep(1)

    except Exception as e:
        fail_list.append((code, name, str(e)))
        print(f"  [失败] {e}")
        import traceback
        traceback.print_exc()


# ── 6. 汇总 ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  完成!")
print(f"{'='*60}")
print(f"  成功: {len(success_list)}/{len(stock_codes)} 只")
print(f"  失败: {len(fail_list)}/{len(stock_codes)} 只")
print(f"")

if success_list:
    total_rows = sum(r for _,_,r in success_list)
    print(f"  成功列表 ({total_rows} 条总记录):")
    for code, name, cnt in success_list:
        print(f"    [OK] {code}  {name}  ({cnt} 天)")

if fail_list:
    print(f"  失败列表:")
    for code, name, err in fail_list:
        print(f"    [FAIL] {code}  {name}  ({err})")

cur.close()
conn.close()
print(f"\nDone.\n")