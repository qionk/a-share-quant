"""
价格预测 - 数据输入
支持 AKShare API 获取 和 Excel 手动上传两种方式
akshare 失败时自动切换到腾讯财经备用源，并尝试通过网易财经补充换手率
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

MAX_RETRIES = 2
RETRY_DELAY = 1


def _market_prefix(stock_code: str) -> str:
    """根据股票代码判断市场前缀"""
    if stock_code.startswith(("6", "9")):
        return "sh"
    return "sz"


def _fetch_via_tencent(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    备用数据源：通过腾讯财经 newfqkline 接口获取前复权日K数据（含换手率）
    """
    import requests as _req

    prefix = _market_prefix(stock_code)
    symbol = f"{prefix}{stock_code}"
    sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    all_klines = []
    current_end = ed
    for _ in range(20):
        # newfqkline 接口：返回含换手率的完整日K
        url = (f"https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
               f"?param={symbol},day,{sd},{current_end},800,qfq")
        try:
            resp = _req.get(url, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
        except Exception:
            # fallback 旧接口
            try:
                url_old = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                           f"?param={symbol},day,{sd},{current_end},800,qfq")
                resp = _req.get(url_old, timeout=10)
                data = resp.json()
            except Exception:
                break

        stock_data = data.get("data", {}).get(symbol, {})
        klines = stock_data.get("qfqday") or stock_data.get("day", [])
        if not klines:
            break

        all_klines = klines + all_klines
        earliest = klines[0][0]
        if earliest <= sd:
            break
        from datetime import datetime as dt, timedelta
        prev = (dt.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        current_end = prev
        time.sleep(0.3)

    if not all_klines:
        return pd.DataFrame()

    # 补齐尾部
    last_fetched = all_klines[-1][0] if all_klines else None
    if last_fetched and last_fetched < ed:
        url = (f"https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
               f"?param={symbol},day,{last_fetched},{ed},30,qfq")
        try:
            resp = _req.get(url, timeout=10)
            if resp.status_code == 200:
                data2 = resp.json()
                klines2 = data2.get("data", {}).get(symbol, {})
                klines2 = klines2.get("qfqday") or klines2.get("day", [])
                if klines2:
                    extra = [k for k in klines2 if k[0] > last_fetched]
                    all_klines.extend(extra)
        except Exception:
            pass

    # 去重（按日期）
    seen = set()
    unique = []
    for k in all_klines:
        if k[0] not in seen:
            seen.add(k[0])
            unique.append(k)
    unique.sort(key=lambda x: x[0])

    # newfqkline 字段: [date, open, close, high, low, volume, {ma}, turnover%, amount, ?]
    # 旧接口字段:      [date, open, close, high, low, volume]
    rows = []
    for row in unique:
        entry = {
            "date": row[0],
            "open": row[1],
            "close": row[2],
            "high": row[3],
            "low": row[4],
            "volume": row[5],
        }
        # newfqkline 有更多字段（第7个是dict/ma，第8个是换手率）
        if len(row) >= 8 and not isinstance(row[7], dict):
            try:
                turnover_val = float(row[7])
                if turnover_val > 0:
                    entry["turnover"] = turnover_val
            except (ValueError, TypeError):
                pass
        if len(row) >= 9:
            try:
                amount_val = float(row[8]) if not isinstance(row[8], dict) else None
                if amount_val and amount_val > 0:
                    entry["amount"] = amount_val * 10000  # 腾讯单位是万元
            except (ValueError, TypeError):
                pass
        rows.append(entry)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算缺失的列
    df["pct_change"] = df["close"].pct_change() * 100
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"] * 100

    return df


def _fetch_turnover_netease(stock_code: str, start_date: str, end_date: str) -> pd.Series:
    """
    通过网易财经接口获取换手率数据，返回 Series(index=DatetimeIndex, values=turnover%)。
    失败返回 None。
    """
    try:
        prefix = "0" + stock_code if stock_code.startswith(("6", "9")) else "1" + stock_code
        sd = f"{start_date[:4]}{start_date[4:6]}{start_date[6:8]}"
        ed = f"{end_date[:4]}{end_date[4:6]}{end_date[6:8]}"
        url = (f"https://quotes.money.163.com/service/chddata.html"
               f"?code={prefix}&start={sd}&end={ed}&fields=TURNOVER")

        content = None
        # 优先用 requests（Streamlit Cloud 无 curl）
        try:
            import requests
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.content:
                content = resp.content.decode("gb2312", errors="ignore")
        except Exception:
            pass
        # fallback: curl
        if not content:
            result = subprocess.run(
                ["curl", "-s", "-m", "15", "-L", url],
                capture_output=True, text=True, timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                content = result.stdout

        if not content:
            return None
        from io import StringIO
        ndf = pd.read_csv(StringIO(content), engine="python")
        if ndf.empty:
            return None
        # 网易列名：日期, 股票代码, 名称, 换手率
        date_col = ndf.columns[0]
        turnover_col = [c for c in ndf.columns if "换手" in c]
        if not turnover_col:
            return None
        ndf[date_col] = pd.to_datetime(ndf[date_col])
        ndf = ndf.sort_values(date_col).set_index(date_col)
        s = pd.to_numeric(ndf[turnover_col[0]], errors="coerce")
        s.index.name = None
        return s
    except Exception:
        return None


def load_from_akshare(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取个股日线数据
    优先用 akshare，失败后切换到东方财富 datacenter 备用源
    """
    # 尝试 akshare（快速失败，尽快 fallback 到腾讯源）
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

                # 检查 AKShare 数据是否覆盖到 end_date，否则用腾讯源补齐最新几天
                end_ts = pd.Timestamp(end_date)
                if df.index[-1] < end_ts:
                    try:
                        # 腾讯源只取 AKShare 缺失的日期段（前复权，价格可能不同，
                        # 但仅用于最新几天的补齐，回测时 pct_change 仍有效）
                        gap_start = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y%m%d")
                        df_gap = _fetch_via_tencent(stock_code, gap_start, end_date)
                        if df_gap is not None and not df_gap.empty:
                            df = pd.concat([df, df_gap]).sort_index()
                            df = df[~df.index.duplicated(keep="last")]
                    except Exception:
                        pass

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
            # 尝试通过网易补充换手率
            turnover_s = _fetch_turnover_netease(stock_code, start_date, end_date)
            if turnover_s is not None and not turnover_s.empty:
                df["turnover"] = turnover_s.reindex(df.index)
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
