"""
波段斐波那契分析（替代短期斐波那契）
使用 scipy.signal.find_peaks 自动检测完整波段高低点
基于波段起点/终点计算标准黄金分割线
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# 斐波那契比率
FIB_LEVELS = [0.0, 0.191, 0.236, 0.382, 0.5, 0.618, 0.786, 0.809, 1.0, 1.272, 1.618, 2.0]
FIB_NAMES = {
    0.0: "波段起点", 0.191: "19.1%", 0.236: "23.6%", 0.382: "38.2%",
    0.5: "50.0%(黄金)", 0.618: "61.8%", 0.786: "78.6%", 0.809: "80.9%",
    1.0: "波段终点", 1.272: "127.2%", 1.618: "161.8%", 2.0: "200.0%",
}
FIB_COLORS = {
    0.0: "blue", 0.191: "lightgreen", 0.236: "orange", 0.382: "green",
    0.5: "gold", 0.618: "darkgreen", 0.786: "darkgreen", 0.809: "green",
    1.0: "red", 1.272: "orange", 1.618: "darkorange", 2.0: "red",
}


def detect_wave_levels(df: pd.DataFrame, wave_window: int = 60,
                       min_wave_return: float = 0.05) -> dict:
    """
    自动检测最近完整波段的高低点。

    参数:
      wave_window: 检测窗口（交易日数）
      min_wave_return: 最小波段涨跌幅（默认5%，小于此幅度视为震荡）

    返回:
      trend: 趋势方向（上涨波段/下跌波段/震荡趋势）
      wave_start/wave_end: 波段起/终点价格
      start_date/end_date: 起/终点日期
      wave_return: 波段累计涨跌幅
      is_valid_wave: 是否为有效波段
    """
    recent_df = df.tail(wave_window).copy()
    close_prices = recent_df['close'].values

    if len(close_prices) < 30:
        recent_high = float(close_prices.max())
        recent_low = float(close_prices.min())
        return {
            'trend': '震荡趋势',
            'wave_start': recent_low,
            'wave_end': recent_high,
            'start_date': recent_df.index[recent_df['close'].idxmin()],
            'end_date': recent_df.index[recent_df['close'].idxmax()],
            'wave_return': (recent_high - recent_low) / recent_low if recent_low > 0 else 0,
            'is_valid_wave': False,
        }

    # 检测局部高低点（距离至少10天）
    peaks, _ = find_peaks(close_prices, distance=10)
    troughs, _ = find_peaks(-close_prices, distance=10)

    if len(peaks) == 0 or len(troughs) == 0:
        recent_high = float(close_prices.max())
        recent_low = float(close_prices.min())
        high_idx = close_prices.argmax()
        low_idx = close_prices.argmin()
        return {
            'trend': '震荡趋势',
            'wave_start': recent_low,
            'wave_end': recent_high,
            'start_date': recent_df.index[low_idx],
            'end_date': recent_df.index[high_idx],
            'wave_return': (recent_high - recent_low) / recent_low if recent_low > 0 else 0,
            'is_valid_wave': False,
        }

    last_peak_idx = peaks[-1]
    last_trough_idx = troughs[-1]
    last_peak_price = float(close_prices[last_peak_idx])
    last_trough_price = float(close_prices[last_trough_idx])
    last_peak_date = recent_df.index[last_peak_idx]
    last_trough_date = recent_df.index[last_trough_idx]

    if last_peak_date > last_trough_date:
        # 高点在低点之后 → 上涨波段
        wave_return = (last_peak_price - last_trough_price) / last_trough_price
        is_valid = wave_return >= min_wave_return
        return {
            'trend': '上涨波段' if is_valid else '震荡趋势',
            'wave_start': last_trough_price,
            'wave_end': last_peak_price,
            'start_date': last_trough_date,
            'end_date': last_peak_date,
            'wave_return': wave_return,
            'is_valid_wave': is_valid,
        }
    else:
        # 低点在高点之后 → 下跌波段
        wave_return = (last_peak_price - last_trough_price) / last_peak_price
        is_valid = abs(wave_return) >= min_wave_return
        return {
            'trend': '下跌波段' if is_valid else '震荡趋势',
            'wave_start': last_peak_price,
            'wave_end': last_trough_price,
            'start_date': last_peak_date,
            'end_date': last_trough_date,
            'wave_return': wave_return,
            'is_valid_wave': is_valid,
        }


def calculate_wave_fibonacci(wave_info: dict) -> list:
    """
    基于波段信息计算所有标准黄金分割价位。

    返回 list[dict]:
      name: 价位名称, price: 价格, type: 类型（支撑/阻力/目标）, color: 显示颜色
    """
    trend = wave_info['trend']
    wave_start = wave_info['wave_start']
    wave_end = wave_info['wave_end']
    wave_range = abs(wave_end - wave_start)
    if wave_range < 0.01 * wave_start:
        wave_range = 0.01 * wave_start

    levels = []

    if trend == "上涨波段":
        # 回撤支撑位（上涨后回调）
        levels.append({'name': '0.191支撑', 'price': round(wave_end - wave_range * 0.191, 2),
                       'type': '弱支撑', 'color': 'lightgreen', 'level': 0.191})
        levels.append({'name': '0.382支撑', 'price': round(wave_end - wave_range * 0.382, 2),
                       'type': '强支撑', 'color': 'green', 'level': 0.382})
        levels.append({'name': '0.5支撑', 'price': round(wave_end - wave_range * 0.5, 2),
                       'type': '黄金支撑', 'color': 'gold', 'level': 0.5})
        levels.append({'name': '0.618支撑', 'price': round(wave_end - wave_range * 0.618, 2),
                       'type': '极强支撑', 'color': 'darkgreen', 'level': 0.618})
        levels.append({'name': '0.809支撑', 'price': round(wave_end - wave_range * 0.809, 2),
                       'type': '强支撑', 'color': 'green', 'level': 0.809})
        # 扩展目标位
        levels.append({'name': '1.272目标', 'price': round(wave_end + wave_range * 0.272, 2),
                       'type': '第一目标', 'color': 'orange', 'level': 1.272})
        levels.append({'name': '1.618目标', 'price': round(wave_end + wave_range * 0.618, 2),
                       'type': '第二目标', 'color': 'darkorange', 'level': 1.618})
        levels.append({'name': '2.0目标', 'price': round(wave_end + wave_range * 1.0, 2),
                       'type': '第三目标', 'color': 'red', 'level': 2.0})
        # 波段高低点
        levels.append({'name': '波段高点', 'price': round(wave_end, 2),
                       'type': '阻力位', 'color': 'red', 'level': 1.0})
        levels.append({'name': '波段低点', 'price': round(wave_start, 2),
                       'type': '强支撑', 'color': 'darkgreen', 'level': 0.0})

    elif trend == "下跌波段":
        # 反弹阻力位（下跌后反弹）
        levels.append({'name': '0.191阻力', 'price': round(wave_end + wave_range * 0.191, 2),
                       'type': '弱阻力', 'color': 'lightcoral', 'level': 0.191})
        levels.append({'name': '0.382阻力', 'price': round(wave_end + wave_range * 0.382, 2),
                       'type': '强阻力', 'color': 'red', 'level': 0.382})
        levels.append({'name': '0.5阻力', 'price': round(wave_end + wave_range * 0.5, 2),
                       'type': '黄金阻力', 'color': 'gold', 'level': 0.5})
        levels.append({'name': '0.618阻力', 'price': round(wave_end + wave_range * 0.618, 2),
                       'type': '极强阻力', 'color': 'darkred', 'level': 0.618})
        levels.append({'name': '0.809阻力', 'price': round(wave_end + wave_range * 0.809, 2),
                       'type': '强阻力', 'color': 'red', 'level': 0.809})
        # 扩展目标位（下跌延续）
        levels.append({'name': '1.272目标', 'price': round(wave_end - wave_range * 0.272, 2),
                       'type': '第一目标', 'color': 'orange', 'level': 1.272})
        levels.append({'name': '1.618目标', 'price': round(wave_end - wave_range * 0.618, 2),
                       'type': '第二目标', 'color': 'darkorange', 'level': 1.618})
        levels.append({'name': '2.0目标', 'price': round(wave_end - wave_range * 1.0, 2),
                       'type': '第三目标', 'color': 'green', 'level': 2.0})
        # 波段高低点
        levels.append({'name': '波段低点', 'price': round(wave_end, 2),
                       'type': '支撑位', 'color': 'green', 'level': 0.0})
        levels.append({'name': '波段高点', 'price': round(wave_start, 2),
                       'type': '强阻力', 'color': 'darkred', 'level': 1.0})

    else:
        # 震荡趋势
        low = min(wave_start, wave_end)
        high = max(wave_start, wave_end)
        levels.append({'name': '震荡下轨', 'price': round(low, 2),
                       'type': '支撑位', 'color': 'green', 'level': 0.0})
        levels.append({'name': '震荡中轨', 'price': round((low + high) / 2, 2),
                       'type': '中性位', 'color': 'gray', 'level': 0.5})
        levels.append({'name': '震荡上轨', 'price': round(high, 2),
                       'type': '阻力位', 'color': 'red', 'level': 1.0})

    levels.sort(key=lambda x: x['price'])
    return levels


def generate_wave_fib_signals(current_price: float, fib_levels: list,
                                wave_info: dict, model_prediction: float = 0,
                                relative_volume: float = 1.0,
                                sensitivity: float = 0.015) -> list:
    """
    基于波段斐波那契价位生成买卖信号。

    参数:
      current_price: 当前价格
      fib_levels: calculate_wave_fibonacci 返回的价位列表
      wave_info: detect_wave_levels 返回的波段信息
      model_prediction: 模型预测的收益率%（正=看涨，负=看跌）
      relative_volume: 相对成交量（与20日均值比）
      sensitivity: 信号灵敏度（默认1.5%，比短期分析更宽松）

    返回 list[dict]: 信号列表
    """
    trend = wave_info['trend']
    is_valid_wave = wave_info['is_valid_wave']
    signals = []

    if not is_valid_wave:
        signals.append({
            'type': '观望',
            'level': '无',
            'price': current_price,
            'reason': '未检测到有效的上涨/下跌波段，建议观望',
            'confidence': 1,
        })
        return signals

    for level_info in fib_levels:
        price_diff = abs(current_price - level_info['price']) / current_price if current_price > 0 else float('inf')

        if price_diff <= sensitivity:
            level_name = level_info['name']
            level_type = level_info['type']
            confidence = 3

            # 结合模型预测调整置信度
            if (trend == "上涨波段" and model_prediction > 0) or \
               (trend == "下跌波段" and model_prediction < 0):
                confidence += 1
            elif (trend == "上涨波段" and model_prediction < 0) or \
                 (trend == "下跌波段" and model_prediction > 0):
                confidence -= 1

            # 结合成交量调整
            if relative_volume > 1.5:
                confidence += 1
            elif relative_volume < 0.7:
                confidence -= 1

            confidence = max(1, min(5, confidence))

            if trend == "上涨波段":
                if "支撑" in level_type:
                    if model_prediction > 0:
                        if "黄金支撑" in level_type or "极强支撑" in level_type:
                            signal_type = '强买入'
                        else:
                            signal_type = '买入'
                        signals.append({
                            'type': signal_type,
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': f'上涨波段回调至{level_name}，量价配合，模型看涨',
                            'confidence': confidence,
                        })
                    else:
                        signals.append({
                            'type': '观望',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': f'上涨波段回调至{level_name}，但模型看跌，建议观望',
                            'confidence': confidence,
                        })
                elif "目标" in level_type:
                    signals.append({
                        'type': '止盈',
                        'level': level_name,
                        'price': level_info['price'],
                        'reason': f'价格到达上涨波段{level_name}，建议分批止盈',
                        'confidence': confidence,
                    })
                elif level_name == "波段高点":
                    if model_prediction > 0:
                        signals.append({
                            'type': '持有/加仓',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': '价格突破波段高点，趋势延续，目标看1.272扩展位',
                            'confidence': confidence,
                        })
                    else:
                        signals.append({
                            'type': '止盈',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': '价格触及波段高点阻力，模型看跌，建议止盈',
                            'confidence': confidence,
                        })

            elif trend == "下跌波段":
                if "阻力" in level_type:
                    if model_prediction < 0:
                        if "黄金阻力" in level_type or "极强阻力" in level_type:
                            signal_type = '强卖出'
                        else:
                            signal_type = '卖出'
                        signals.append({
                            'type': signal_type,
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': f'下跌波段反弹至{level_name}，量价配合，模型看跌',
                            'confidence': confidence,
                        })
                    else:
                        signals.append({
                            'type': '观望',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': f'下跌波段反弹至{level_name}，但模型看涨，建议观望',
                            'confidence': confidence,
                        })
                elif "目标" in level_type:
                    signals.append({
                        'type': '轻仓抄底',
                        'level': level_name,
                        'price': level_info['price'],
                        'reason': f'价格到达下跌波段{level_name}，可轻仓尝试抄底',
                        'confidence': confidence,
                    })
                elif level_name == "波段低点":
                    if model_prediction < 0:
                        signals.append({
                            'type': '止损/清仓',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': '价格跌穿波段低点，趋势延续，目标看1.272扩展位',
                            'confidence': confidence,
                        })
                    else:
                        signals.append({
                            'type': '轻仓抄底',
                            'level': level_name,
                            'price': level_info['price'],
                            'reason': '价格触及波段低点支撑，模型看涨，可轻仓抄底',
                            'confidence': confidence,
                        })

    # 如果没有匹配信号，给出默认建议
    if not signals:
        signals.append({
            'type': '持有/观望',
            'level': '无',
            'price': current_price,
            'reason': '当前价格未接近任何关键黄金分割位，继续持有或观望',
            'confidence': 2,
        })

    return signals