"""
股票日线数据 MySQL 存取
store: AKShare → MySQL
load:  MySQL → DataFrame
list:  所有已存储股票概览
"""

import os
import time as _time
import pandas as pd
import numpy as np
from datetime import datetime

BATCH_SIZE = 500
_TABLE_ENSURED = False


def _get_conn():
    try:
        import streamlit as st
        host = st.secrets.get("MYSQL_HOST", os.environ.get("MYSQL_HOST", ""))
        port = int(st.secrets.get("MYSQL_PORT", os.environ.get("MYSQL_PORT", "3306")))
        user = st.secrets.get("MYSQL_USER", os.environ.get("MYSQL_USER", ""))
        password = st.secrets.get("MYSQL_PASSWORD", os.environ.get("MYSQL_PASSWORD", ""))
        database = st.secrets.get("MYSQL_DATABASE", os.environ.get("MYSQL_DATABASE", ""))
    except Exception:
        host = os.environ.get("MYSQL_HOST", "")
        port = int(os.environ.get("MYSQL_PORT", "3306"))
        user = os.environ.get("MYSQL_USER", "")
        password = os.environ.get("MYSQL_PASSWORD", "")
        database = os.environ.get("MYSQL_DATABASE", "")

    if not host or not user or not database:
        return None

    import pymysql
    return pymysql.connect(
        host=host, port=port, user=user, password=password,
        database=database, charset="utf8mb4",
        connect_timeout=10, read_timeout=30, write_timeout=30,
        autocommit=True,
    )


def _ensure_unique_index(conn):
    """确保 stock_daily_data 有 (stock_code, trade_date) 唯一索引，只执行一次"""
    global _TABLE_ENSURED
    if _TABLE_ENSURED:
        return
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                stock_code VARCHAR(20) NOT NULL,
                stock_name VARCHAR(100),
                trade_date DATE NOT NULL,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
                volume DOUBLE, amount DOUBLE, pct_change DOUBLE, turnover DOUBLE,
                UNIQUE KEY uk_stock_date (stock_code, trade_date),
                INDEX idx_stock_code (stock_code)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    except Exception:
        pass
    # 检查唯一索引是否已存在
    cur.execute("SHOW INDEX FROM stock_daily_data WHERE Key_name='uk_stock_date'")
    if not cur.fetchall():
        # 索引不存在，先去重再加
        try:
            cur.execute("""
                DELETE d1 FROM stock_daily_data d1
                INNER JOIN stock_daily_data d2
                WHERE d1.id < d2.id
                  AND d1.stock_code = d2.stock_code
                  AND d1.trade_date = d2.trade_date
            """)
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE stock_daily_data ADD UNIQUE KEY uk_stock_date (stock_code, trade_date)")
        except Exception:
            pass
    cur.close()
    _TABLE_ENSURED = True


_DEDUP_DONE = False


def _dedup_stock_table():
    """清理 stock_daily_data 中的重复行并确保唯一索引，整个进程只执行一次"""
    global _DEDUP_DONE
    if _DEDUP_DONE:
        return
    conn = _get_conn()
    if not conn:
        return
    _ensure_unique_index(conn)
    conn.close()
    _DEDUP_DONE = True


def store_stock_data(stock_code: str, stock_name: str, df: pd.DataFrame) -> int:
    """将 DataFrame 写入 stock_daily_data，返回写入行数"""
    conn = _get_conn()
    if not conn:
        return 0

    _ensure_unique_index(conn)
    cur = conn.cursor()
    sql = """
        INSERT INTO stock_daily_data
        (stock_code, stock_name, trade_date, open, high, low, close, volume, amount, pct_change, turnover)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close),
        volume=VALUES(volume), amount=VALUES(amount), pct_change=VALUES(pct_change),
        turnover=VALUES(turnover), stock_name=VALUES(stock_name)
    """

    rows = []
    for idx, row in df.iterrows():
        trade_date = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
        rows.append((
            stock_code, stock_name, trade_date,
            _safe_float(row.get("open")),
            _safe_float(row.get("high")),
            _safe_float(row.get("low")),
            _safe_float(row.get("close")),
            _safe_float(row.get("volume")),
            _safe_float(row.get("amount")),
            _safe_float(row.get("pct_change")),
            _safe_float(row.get("turnover")),
        ))

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        cur.executemany(sql, batch)

    cur.close()
    conn.close()
    return len(rows)


def _trim_stock_data(stock_code: str, keep_days: int):
    """删除某股票超过 keep_days 天的旧数据"""
    conn = _get_conn()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute(
        "SELECT trade_date FROM stock_daily_data WHERE stock_code=%s "
        "ORDER BY trade_date DESC LIMIT 1 OFFSET %s",
        (stock_code, keep_days),
    )
    row = cur.fetchone()
    if row:
        cutoff = row[0]
        cur.execute(
            "DELETE FROM stock_daily_data WHERE stock_code=%s AND trade_date < %s",
            (stock_code, cutoff),
        )
    cur.close()
    conn.close()


def delete_stock_data(stock_code: str):
    """删除某股票的全部数据（用于重新获取全量数据）"""
    conn = _get_conn()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("DELETE FROM stock_daily_data WHERE stock_code=%s", (stock_code,))
    cur.close()
    conn.close()


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def load_stock_from_db(stock_code: str) -> pd.DataFrame:
    """从 MySQL 加载股票日线数据，返回 DataFrame（date 索引，去重）"""
    conn = _get_conn()
    if not conn:
        return pd.DataFrame()

    cur = conn.cursor()
    cur.execute(
        "SELECT trade_date, open, high, low, close, volume, amount, pct_change, turnover "
        "FROM stock_daily_data WHERE stock_code=%s ORDER BY trade_date",
        (stock_code,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df


def list_db_stocks() -> list:
    """返回已存储的股票列表 [{code, name, rows, start_date, end_date}]，数据量受 max_days 上限约束"""
    conn = _get_conn()
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute("""
        SELECT stock_code, stock_name, COUNT(DISTINCT trade_date) AS data_rows,
               MIN(trade_date) AS start_date, MAX(trade_date) AS end_date
        FROM stock_daily_data
        GROUP BY stock_code, stock_name
        ORDER BY stock_code
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["rows"] = r.pop("data_rows")
    for r in rows:
        r["code"] = r["stock_code"]
        r["name"] = r["stock_name"]
        r["start_date"] = r["start_date"].strftime("%Y-%m-%d") if hasattr(r["start_date"], "strftime") else str(r["start_date"])
        r["end_date"] = r["end_date"].strftime("%Y-%m-%d") if hasattr(r["end_date"], "strftime") else str(r["end_date"])
    cur.close()
    conn.close()
    return rows


def list_stocks_with_status() -> list:
    """统一视图: 股票数据 + 训练状态 [{code, name, data_rows, start_date, end_date,
       trained, trained_at, trained_models, session_id}]"""
    _dedup_stock_table()
    conn = _get_conn()
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute("""
        SELECT d.stock_code, d.stock_name,
               COUNT(DISTINCT d.trade_date) AS data_rows,
               MIN(d.trade_date) AS start_date,
               MAX(d.trade_date) AS end_date,
               MAX(t.trained_at) AS trained_at,
               MAX(t.selected_models) AS trained_models,
               MAX(t.id) AS session_id
        FROM stock_daily_data d
        LEFT JOIN training_sessions t ON d.stock_code = t.stock_code
        GROUP BY d.stock_code, d.stock_name
        ORDER BY d.stock_code
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["code"] = r["stock_code"]
        r["name"] = r["stock_name"]
        r["rows"] = r.get("data_rows", 0)
        r["trained"] = r["trained_at"] is not None
        r["start_date"] = r["start_date"].strftime("%Y-%m-%d") if hasattr(r["start_date"], "strftime") else str(r["start_date"])
        r["end_date"] = r["end_date"].strftime("%Y-%m-%d") if hasattr(r["end_date"], "strftime") else str(r["end_date"])
        if r["trained_at"] and hasattr(r["trained_at"], "strftime"):
            r["trained_at"] = r["trained_at"].strftime("%Y-%m-%d %H:%M")
        if r.get("trained_models"):
            import json
            r["trained_models"] = json.loads(r["trained_models"]) if isinstance(r["trained_models"], str) else r["trained_models"]
    cur.close()
    conn.close()
    return rows


def has_stock_data(stock_code: str) -> bool:
    conn = _get_conn()
    if not conn:
        return False
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM stock_daily_data WHERE stock_code=%s LIMIT 1", (stock_code,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists


def get_stock_name_from_db(stock_code: str) -> str:
    conn = _get_conn()
    if not conn:
        return ""
    cur = conn.cursor()
    cur.execute("SELECT stock_name FROM stock_daily_data WHERE stock_code=%s LIMIT 1", (stock_code,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else ""


def list_stock_sessions(stock_code: str) -> list:
    """返回某股票的所有训练记录 [{session_id, trained_at, trained_models, forecast_days}]"""
    conn = _get_conn()
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT id, trained_at, selected_models, forecast_days "
        "FROM training_sessions WHERE stock_code=%s ORDER BY trained_at DESC",
        (stock_code,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["session_id"] = r["id"]
        if hasattr(r["trained_at"], "strftime"):
            r["trained_at"] = r["trained_at"].strftime("%Y-%m-%d %H:%M")
        if r.get("selected_models"):
            import json
            r["trained_models"] = json.loads(r["selected_models"]) if isinstance(r["selected_models"], str) else r["selected_models"]
    cur.close()
    conn.close()
    return rows


def _fill_turnover_if_missing(stock_code: str, stock_name: str, df: pd.DataFrame,
                              start_date: str, end_date: str) -> pd.DataFrame:
    """如果 turnover 缺失超过 50%，从网易源补充并批量更新 DB"""
    if 'turnover' not in df.columns or df.empty:
        return df
    valid_ratio = df['turnover'].notna().mean()
    if valid_ratio >= 0.5:
        return df
    try:
        from src.predict.data_input import _fetch_turnover_netease
        turnover_s = _fetch_turnover_netease(stock_code, start_date, end_date)
        if turnover_s is not None and not turnover_s.empty:
            df['turnover'] = turnover_s.reindex(df.index)
            # 批量写回 DB
            conn = _get_conn()
            if conn:
                cur = conn.cursor()
                updates = []
                for idx in df.index:
                    t_val = _safe_float(df.loc[idx, 'turnover'])
                    if t_val is not None:
                        trade_date = idx.date() if hasattr(idx, 'date') else pd.Timestamp(idx).date()
                        updates.append((t_val, stock_code, trade_date))
                if updates:
                    cur.executemany(
                        "UPDATE stock_daily_data SET turnover=%s "
                        "WHERE stock_code=%s AND trade_date=%s",
                        updates)
                cur.close()
                conn.close()
    except Exception:
        pass
    return df


def fetch_and_store(stock_code: str, start_date: str = "20200101",
                    end_date: str = None, max_days: int = 500,
                    progress_callback=None) -> tuple:
    """
    从 AKShare 获取数据并存入 MySQL
    start_date/end_date: YYYYMMDD 格式
    max_days: 最多保留最近多少交易日（None=不限制，建议500）
    返回: (DataFrame, stock_name, 是否已有数据)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 首次调用时确保唯一索引 + 清理历史重复数据
    _dedup_stock_table()

    if progress_callback:
        progress_callback("checking")

    # 已存在则检查是否需要重新获取
    if has_stock_data(stock_code):
        df = load_stock_from_db(stock_code)
        name = get_stock_name_from_db(stock_code)

        if df.empty:
            # 空数据，走重新获取流程
            pass
        else:
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            db_start = df.index[0].strftime("%Y%m%d") if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10].replace("-", "")

            # 若请求的起始日期早于DB中最早日期，需要重新获取全量数据
            if start_date < db_start:
                if progress_callback:
                    progress_callback("refetching")
                delete_stock_data(stock_code)
                # 跳出 if 块，走下面的全量获取
            else:
                # 已有数据在请求范围内，仅增量更新
                if last_date < today:
                    if progress_callback:
                        progress_callback("updating")
                    from src.predict.data_input import load_from_akshare
                    new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    new_end = end_date or datetime.now().strftime("%Y%m%d")
                    try:
                        df_new = load_from_akshare(stock_code, new_start, new_end)
                        if df_new is not None and len(df_new) > 0:
                            store_stock_data(stock_code, name, df_new)
                            df = pd.concat([df, df_new]).sort_index()
                            df = df[~df.index.duplicated(keep="last")]
                    except Exception as e:
                        if progress_callback:
                            progress_callback("update_failed")

                # 按请求的起始日期截取
                req_start = pd.Timestamp(start_date)
                df = df[df.index >= req_start]

                # 补充换手率：如果 DB 数据中 turnover 大部分为空，尝试网易源补齐
                df = _fill_turnover_if_missing(stock_code, name, df, start_date, end_date)

                # 按 max_days 截取
                if max_days and len(df) > max_days:
                    df = df.tail(max_days)
                    _trim_stock_data(stock_code, max_days)
                if progress_callback:
                    progress_callback("done")
                return df, name, True

    if progress_callback:
        progress_callback("fetching")

    # AKShare 获取
    from src.predict.data_input import load_from_akshare, get_stock_name
    df = load_from_akshare(stock_code, start_date, end_date)
    name = get_stock_name(stock_code)

    # 截取最近 max_days 天
    if max_days and len(df) > max_days:
        df = df.tail(max_days)

    if progress_callback:
        progress_callback("storing")

    store_stock_data(stock_code, name, df)

    # 首次获取后也清理超出限制的旧数据
    if max_days:
        _trim_stock_data(stock_code, max_days)

    if progress_callback:
        progress_callback("done")

    return df, name, False