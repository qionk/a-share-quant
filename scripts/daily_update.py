#!/usr/bin/env python3
"""
每日增量更新：检查数据库中所有股票，自动拉取缺失的交易日数据

用法:
  # 本地运行（需配好 .streamlit/secrets.toml）
  python scripts/daily_update.py

  # 通过环境变量指定数据库
  MYSQL_HOST=xxx MYSQL_PORT=3308 MYSQL_USER=xxx MYSQL_PASSWORD=xxx MYSQL_DATABASE=xxx python scripts/daily_update.py

  # GitHub Actions 示例:
  # - cron: '0 10 * * 1-5'  # 工作日每天早上10点
  # - env: MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE via Secrets
"""
import sys, os, time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymysql
import pandas as pd
import numpy as np


def get_conn():
    host = os.environ.get("MYSQL_HOST", "")
    port = int(os.environ.get("MYSQL_PORT", "3306"))
    user = os.environ.get("MYSQL_USER", "")
    password = os.environ.get("MYSQL_PASSWORD", "")
    database = os.environ.get("MYSQL_DATABASE", "")

    if not host or not user or not database:
        # fallback: read from .streamlit/secrets.toml
        try:
            import tomllib
            secrets_path = os.path.join(ROOT, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
                host = secrets.get("MYSQL_HOST", "")
                port = int(secrets.get("MYSQL_PORT", "3306"))
                user = secrets.get("MYSQL_USER", "")
                password = secrets.get("MYSQL_PASSWORD", "")
                database = secrets.get("MYSQL_DATABASE", "")
        except Exception:
            pass

    if not host or not user or not database:
        print("[错误] MySQL 连接信息未配置!")
        print("请设置环境变量: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE")
        print("或在 .streamlit/secrets.toml 中配置")
        sys.exit(1)

    return pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset="utf8mb4",
        connect_timeout=10, read_timeout=30, write_timeout=30,
        autocommit=True,
    )


def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def main():
    print(f"\n{'='*60}")
    print(f"  每日增量数据更新")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    conn = get_conn()
    cur = conn.cursor()
    print("\n[OK] MySQL 连接成功")

    # 1. 列出所有股票及其最新日期
    cur.execute("""
        SELECT stock_code, stock_name, MAX(trade_date) AS last_date
        FROM stock_daily_data
        GROUP BY stock_code, stock_name
        ORDER BY stock_code
    """)
    stocks = [(r[0], r[1], r[2]) for r in cur.fetchall()]

    if not stocks:
        print("数据库中没有股票，无需更新")
        cur.close()
        conn.close()
        return

    print(f"\n共 {len(stocks)} 只股票待检查\n")

    # 2. 导入 AKShare
    from src.predict.data_input import load_from_akshare

    INSERT_SQL = """
        INSERT INTO stock_daily_data
        (stock_code, stock_name, trade_date, open, high, low, close, volume, amount, pct_change, turnover)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close),
        volume=VALUES(volume), amount=VALUES(amount), pct_change=VALUES(pct_change),
        turnover=VALUES(turnover), stock_name=VALUES(stock_name)
    """

    today = datetime.now()
    updated = 0
    skipped = 0
    failed = 0

    for code, name, last_date in stocks:
        # 判断是否需要更新：最新日期 < 今天（允许1天延迟，A股数据次日更新）
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
        elif hasattr(last_date, "date"):
            last_date = last_date.date()

        days_behind = (today.date() - last_date).days
        # 周末/假期没有交易数据，days_behind == 0 表示已有当天数据才跳过
        if days_behind == 0:
            skipped += 1
            continue

        try:
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            print(f"[更新] {code} {name}  缺 {days_behind} 天  ({last_date} → {today.date()})")

            df = load_from_akshare(code, start, end)
            if df is None or len(df) == 0:
                skipped += 1
                continue

            rows = []
            for idx, row in df.iterrows():
                trade_date = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
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

            for i in range(0, len(rows), 500):
                cur.executemany(INSERT_SQL, rows[i:i + 500])

            updated += 1
            print(f"  [OK] +{len(rows)} 条")

            time.sleep(1)  # 反爬间隔

        except Exception as e:
            failed += 1
            print(f"  [失败] {e}")

    # 3. 汇总
    print(f"\n{'='*60}")
    print(f"  完成!  更新: {updated}  跳过(已有最新): {skipped}  失败: {failed}")
    print(f"{'='*60}\n")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()