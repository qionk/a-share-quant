"""
价格预测 - 数据输入
支持 AKShare API 获取 和 Excel 手动上传两种方式
akshare 失败时自动切换到东方财富 datacenter API 备用源
"""

import io
import time
import json
import subprocess
import pandas as pd
import numpy as np
import akshare as ak

REQUIRED_COLUMNS = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "涨跌幅"]

COLUMN_MAP = {
    "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change",
}

MAX_RETRIES = 3
RETRY_DELAY = 2


def _market_prefix(stock_code: str) -> str:
    """根据股票代码判断市场前缀"""
    if stock_code.startswith(("6", "9")):
        return "sh"
    return "sz"


def _fetch_via_tencent(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    备用数据源：通过腾讯财经接口获取前复权日K数据
    """
    prefix = _market_prefix(stock_code)
    symbol = f"{prefix}{stock_code}"
    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    all_klines = []
    # 腾讯接口每次最多返回约300条，需要分段请求
    current_end = ed
    for _ in range(20):
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={symbol},day,{sd},{current_end},800,qfq")
        result = subprocess.run(
            ["curl", "-s", "-m", "10", url],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            break
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            break

        stock_data = data.get("data", {}).get(symbol, {})
        klines = stock_data.get("qfqday") or stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines
        earliest = klines[0][0]
        if earliest <= sd:
            break
        # 下一段的结束日期为当前最早日期的前一天
        from datetime import datetime as dt, timedelta
        prev = (dt.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        current_end = prev
        time.sleep(0.3)

    if not all_klines:
        return pd.DataFrame()

    # 去重（按日期）
    seen = set()
    unique = []
    for k in all_klines:
        if k[0] not in seen:
            seen.add(k[0])
            unique.append(k)
    unique.sort(key=lambda x: x[0])

    # 统一截取前6列（date, open, close, high, low, volume），忽略可能存在的第7列
    trimmed = [row[:6] for row in unique]
    df = pd.DataFrame(trimmed, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算缺失的列
    df["pct_change"] = df["close"].pct_change() * 100
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"] * 100  # 估算

    return df


def load_from_akshare(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取个股日线数据
    优先用 akshare，失败后切换到东方财富 datacenter 备用源
    """
    # 尝试 akshare
    df = None
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code, period="daily",
                start_date=start_date, end_date=end_date,
                adjust="hfq",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "涨跌幅": "pct_change", "换手率": "turnover",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")
                needed = ["open", "high", "low", "close", "volume", "amount", "pct_change"]
                for col in needed:
                    if col not in df.columns:
                        if col == "pct_change":
                            df["pct_change"] = df["close"].pct_change() * 100
                return df
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    # akshare 失败，切换腾讯财经备用源
    try:
        df = _fetch_via_tencent(stock_code, start_date, end_date)
        if df is not None and not df.empty:
            if "pct_change" not in df.columns or df["pct_change"].isna().all():
                df["pct_change"] = df["close"].pct_change() * 100
            return df
    except Exception:
        pass

    msg = f"未获取到 {stock_code} 的数据（主备数据源均失败）"
    if last_err:
        msg += f"\n原始错误: {type(last_err).__name__}: {last_err}"
    raise ValueError(msg)


STOCK_NAME_CACHE = {
    "601869": "长飞光纤",
    "603601": "再升科技",
    "601138": "工业富联",
}


def get_stock_name(stock_code: str) -> str:
    """查询股票名称（优先用缓存，失败时回退到代码）"""
    if stock_code in STOCK_NAME_CACHE:
        return STOCK_NAME_CACHE[stock_code]
    try:
        info = ak.stock_info_a_code_name()
        info.columns = ["code", "name"]
        match = info[info["code"] == stock_code]
        if not match.empty:
            name = match.iloc[0]["name"]
            STOCK_NAME_CACHE[stock_code] = name
            return name
    except Exception:
        pass
    return stock_code


def validate_dataframe(df: pd.DataFrame) -> tuple:
    """
    验证上传的 DataFrame 格式
    返回: (是否通过, 错误信息列表)
    """
    errors = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"缺少必需列: {', '.join(missing)}")
        return False, errors

    try:
        pd.to_datetime(df["日期"])
    except Exception:
        errors.append("'日期' 列格式无法解析，请使用 YYYY-MM-DD 或 YYYYMMDD 格式")

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "日期"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_numeric(df[col])
            except Exception:
                errors.append(f"'{col}' 列包含非数值数据")

    if len(df) < 30:
        errors.append(f"数据行数不足: {len(df)} 行（至少需要 30 行）")

    na_counts = df[REQUIRED_COLUMNS].isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    if not cols_with_na.empty:
        for col, cnt in cols_with_na.items():
            errors.append(f"'{col}' 列有 {cnt} 个缺失值")

    return len(errors) == 0, errors


def load_from_excel(uploaded_file) -> pd.DataFrame:
    """
    读取上传的 Excel 文件并标准化
    返回标准化 DataFrame（date 为 DatetimeIndex）
    """
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            df = pd.read_excel(uploaded_file, engine=None)
            break
        except Exception:
            uploaded_file.seek(0)
            continue
    else:
        raise ValueError("无法读取 Excel 文件，请确认文件格式正确")

    ok, errors = validate_dataframe(df)
    if not ok:
        raise ValueError("Excel 数据验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

    result = pd.DataFrame()
    for cn_col, en_col in COLUMN_MAP.items():
        if cn_col in df.columns:
            result[en_col] = df[cn_col]

    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").set_index("date")

    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["close"])
    return result


def generate_template() -> bytes:
    """生成可下载的 Excel 模板（含示例数据和说明）"""
    dates = pd.bdate_range("2024-01-02", periods=5)
    sample = pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": [10.50, 10.80, 10.60, 10.90, 11.00],
        "最高": [10.90, 10.95, 10.85, 11.10, 11.20],
        "最低": [10.40, 10.55, 10.50, 10.75, 10.90],
        "收盘": [10.80, 10.60, 10.80, 11.05, 11.10],
        "成交量": [500000, 450000, 520000, 600000, 550000],
        "成交额": [5300000, 4800000, 5500000, 6500000, 6100000],
        "涨跌幅": [1.50, -1.85, 1.89, 2.31, 0.45],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        sample.to_excel(writer, sheet_name="日线数据", index=False)

        instructions = pd.DataFrame({
            "说明": [
                "请按照'日线数据'工作表的格式填入数据",
                "日期格式: YYYY-MM-DD",
                "涨跌幅单位: %（如 1.50 表示涨 1.50%）",
                "成交量单位: 股",
                "成交额单位: 元",
                "所有列均为必填项",
                "至少需要 30 个交易日的数据",
            ]
        })
        instructions.to_excel(writer, sheet_name="填写说明", index=False)

    buf.seek(0)
    return buf.getvalue()
