"""
A股量化模型 - 数据层
AKShare 数据获取 + DuckDB 本地缓存
"""

import os
import time
import duckdb
import pandas as pd
import akshare as ak
import yaml
from datetime import datetime, timedelta
from tqdm import tqdm


def load_config(config_path=None):
    """加载配置文件，自动查找项目根目录"""
    if config_path is None:
        # 从当前文件向上查找 config.yaml
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 将相对路径转为基于项目根目录的绝对路径
    base = os.path.dirname(config_path)
    cfg["data"]["db_path"] = os.path.join(base, cfg["data"]["db_path"])
    return cfg


def _get_db(db_path, read_only=False):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path, read_only=read_only)


def init_db(db_path):
    """初始化 DuckDB 表结构"""
    con = _get_db(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            code VARCHAR PRIMARY KEY,
            name VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            code VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            amount DOUBLE,
            pct_change DOUBLE,
            turnover DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_prices (
            code VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            amount DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    con.close()


# ── AKShare 数据获取 ──────────────────────────────────────────

def fetch_stock_list():
    """获取全 A 股代码和名称"""
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    return df


def fetch_stock_hist(symbol, start_date, end_date, adjust="hfq"):
    """获取单只股票后复权日线数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "涨跌幅": "pct_change", "换手率": "turnover",
        })
        df["code"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        cols = ["code", "date", "open", "high", "low", "close",
                "volume", "amount", "pct_change", "turnover"]
        return df[[c for c in cols if c in df.columns]]
    except Exception as e:
        return None


def fetch_index_hist(symbol, start_date, end_date):
    """获取指数日线数据"""
    try:
        df = ak.stock_zh_index_daily_em(
            symbol=symbol, start_date=start_date, end_date=end_date,
        )
        if df is None or df.empty:
            return None
        df["code"] = symbol
        df["date"] = pd.to_datetime(df["date"])
        return df[["code", "date", "open", "high", "low", "close", "volume", "amount"]]
    except Exception as e:
        print(f"  [WARN] 指数 {symbol} 获取失败: {e}")
        return None


# ── 数据更新 ──────────────────────────────────────────────────

def update_all_data(config):
    """拉取/增量更新全部数据到 DuckDB"""
    db_path = config["data"]["db_path"]
    start_date = config["data"]["start_date"]
    end_date = datetime.now().strftime("%Y%m%d")

    init_db(db_path)
    con = _get_db(db_path)

    # 1. 股票列表
    print("[1/3] 获取股票列表...")
    stock_list = fetch_stock_list()
    con.execute("DELETE FROM stocks")
    con.execute("INSERT INTO stocks SELECT * FROM stock_list")
    print(f"      共 {len(stock_list)} 只股票")

    # 2. 已有数据 → 增量更新
    existing = con.execute(
        "SELECT code, MAX(date) as last_date FROM daily_prices GROUP BY code"
    ).fetchdf()
    existing_dict = (
        dict(zip(existing["code"], existing["last_date"]))
        if not existing.empty else {}
    )

    print("[2/3] 更新日线数据...")
    failed = []
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), ncols=80):
        code = row["code"]
        if code in existing_dict and existing_dict[code] is not None:
            s = (pd.to_datetime(existing_dict[code]) + timedelta(days=1)).strftime("%Y%m%d")
        else:
            s = start_date

        if s > end_date:
            continue

        df = fetch_stock_hist(code, s, end_date)
        if df is not None and not df.empty:
            con.execute("INSERT OR REPLACE INTO daily_prices SELECT * FROM df")
        else:
            failed.append(code)

        time.sleep(0.25)  # 限速

    if failed:
        print(f"      {len(failed)} 只股票获取失败（可稍后重试）")

    # 3. 指数
    print("[3/3] 更新指数数据...")
    index_code = config["market"]["index_code"]
    idx_df = fetch_index_hist(index_code, start_date, end_date)
    if idx_df is not None:
        con.execute(f"DELETE FROM index_prices WHERE code = '{index_code}'")
        con.execute("INSERT INTO index_prices SELECT * FROM idx_df")

    con.close()
    print("数据更新完成")


# ── 数据加载 ──────────────────────────────────────────────────

def get_stock_pool(config):
    """根据条件筛选股票池"""
    con = _get_db(config["data"]["db_path"], read_only=True)
    pool_cfg = config["pool"]

    stocks = con.execute("SELECT code, name FROM stocks").fetchdf()
    if pool_cfg["exclude_st"]:
        stocks = stocks[~stocks["name"].str.contains("ST", case=False, na=False)]

    min_days = pool_cfg["min_list_days"]
    min_amount = pool_cfg["min_daily_amount"]
    min_price = pool_cfg["exclude_price_below"]

    qualified = con.execute(f"""
        SELECT code,
               COUNT(*) as trading_days,
               AVG(amount) as avg_amount,
               MIN(close) as min_close
        FROM daily_prices
        WHERE date >= CURRENT_DATE - INTERVAL '{min_days * 2} days'
        GROUP BY code
        HAVING trading_days >= {min_days}
           AND avg_amount >= {min_amount}
           AND min_close >= {min_price}
    """).fetchdf()

    pool = stocks[stocks["code"].isin(qualified["code"])].reset_index(drop=True)
    con.close()
    return pool


def load_panel_data(config, codes=None, start_date=None):
    """
    加载面板数据，返回 dict of pivot DataFrames:
    {"close": date×code, "volume": date×code, ...}
    """
    con = _get_db(config["data"]["db_path"], read_only=True)

    query = ("SELECT code, date, open, high, low, close, "
             "volume, amount, pct_change, turnover FROM daily_prices")
    conds = []
    if codes:
        code_str = ",".join(f"'{c}'" for c in codes)
        conds.append(f"code IN ({code_str})")
    if start_date:
        conds.append(f"date >= '{start_date}'")
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY date"

    df = con.execute(query).fetchdf()
    con.close()

    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    panel = {}
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover"]:
        if col in df.columns:
            panel[col] = df.pivot(index="date", columns="code", values=col)
    return panel


def load_index_data(config):
    """加载基准指数数据"""
    con = _get_db(config["data"]["db_path"], read_only=True)
    idx = config["market"]["index_code"]
    df = con.execute(
        "SELECT date, open, high, low, close, volume, amount "
        "FROM index_prices WHERE code = ? ORDER BY date", [idx]
    ).fetchdf()
    con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")
