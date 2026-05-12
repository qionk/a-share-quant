"""
GARCH(1,1) 波动率预测模型（替代原 EGARCH）
输入：日收益率序列（百分比形式）
输出：未来波动率、风险指标
"""

import numpy as np
import warnings


def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1, dist: str = 't'):
    """
    拟合 GARCH(1,1) 模型。

    returns: 日收益率序列（百分比，范围约 ±10）
    mean='constant' 允许非零均值收益
    dist='t' 捕获厚尾特性
    """
    from arch import arch_model

    warnings.filterwarnings("ignore")
    try:
        model = arch_model(returns, vol='GARCH', p=p, q=q, mean='constant', dist=dist)
        result = model.fit(disp='off', show_warning=False)
        return result
    except Exception:
        return None


def predict_garch_volatility(model_result, horizon: int = 5) -> dict:
    """
    使用已拟合的 GARCH 模型预测未来 horizon 天的波动率。

    返回:
      - mean_forecast: 均值预测（日收益率的点估计）
      - volatility_forecast: 波动率预测（标准差）
      - confidence_lower: 95% 下置信界
      - confidence_upper: 95% 上置信界
    """
    forecast = model_result.forecast(horizon=horizon)
    mean_fc = forecast.mean.iloc[-1].values
    var_fc = forecast.variance.iloc[-1].values
    vol_fc = np.sqrt(np.maximum(var_fc, 0))

    return {
        'mean_forecast': mean_fc,
        'volatility_forecast': vol_fc,
        'confidence_lower': mean_fc - 1.96 * vol_fc,
        'confidence_upper': mean_fc + 1.96 * vol_fc,
    }


def compute_risk_metrics(returns: np.ndarray, garch_result=None) -> dict:
    """
    计算风险指标。

    返回:
      - annual_vol_pct: 年化波动率（%）
      - var_95_pct: 95% 置信度 VaR（%）
      - cvar_95_pct: 95% 置信度 CVaR（%）
      - risk_level: 风险等级 (低/中/高)
    """
    recent = returns[-60:] if len(returns) >= 60 else returns
    annual_vol = float(np.std(recent) * np.sqrt(252))
    var_95 = float(np.percentile(recent, 5))
    cvar_95 = float(recent[recent <= var_95].mean()) if (recent <= var_95).any() else var_95

    if annual_vol < 20:
        risk_level = '低'
    elif annual_vol < 40:
        risk_level = '中'
    else:
        risk_level = '高'

    metrics = {
        'annual_vol_pct': round(annual_vol, 2),
        'var_95_pct': round(var_95, 2),
        'cvar_95_pct': round(cvar_95, 2),
        'risk_level': risk_level,
    }

    if garch_result is not None:
        try:
            cond_vol = float(np.sqrt(garch_result.conditional_volatility[-1]))
            metrics['garch_cond_vol_pct'] = round(cond_vol, 2)
        except Exception:
            pass

    return metrics


def historical_volatility_fallback(returns: np.ndarray, horizon: int = 5) -> dict:
    """当 GARCH 拟合失败时使用历史波动率作为替代"""
    vol = float(np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns))
    mean = float(np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns))
    mean_arr = np.full(horizon, mean)
    vol_arr = np.full(horizon, vol)
    return {
        'mean_forecast': mean_arr,
        'volatility_forecast': vol_arr,
        'confidence_lower': mean_arr - 1.96 * vol_arr,
        'confidence_upper': mean_arr + 1.96 * vol_arr,
    }