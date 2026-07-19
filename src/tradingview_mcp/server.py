"""
TradingView MCP Server — routing layer only.

Each @mcp.tool() handler is responsible for:
  1. Validating / sanitising parameters
  2. Delegating to the appropriate service module
  3. Returning the result

No business logic lives here. All computation is in core/services/*.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings  # <-- Ditambahkan
from mcp.types import ToolAnnotations

# ── Service imports ────────────────────────────────────────────────────────────
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.screener_service import (
    fetch_bollinger_analysis,
    fetch_trending_analysis,
    analyze_coin,
    scan_consecutive_candles,
    scan_advanced_candle_patterns_single_tf,
    fetch_multi_timeframe_patterns,
    run_multi_timeframe_analysis,
)
from tradingview_mcp.core.services.scanner_service import (
    volume_breakout_scan,
    volume_confirmation_analyze,
    smart_volume_scan,
)
from tradingview_mcp.core.services.multi_agent_service import run_multi_agent_analysis
from tradingview_mcp.core.services.egx_service import (
    get_egx_market_overview,
    scan_egx_sector,
    run_egx_sector_scanner,
    analyze_egx_index,
    screen_egx_stocks,
    generate_egx_trade_plan,
    analyze_egx_fibonacci,
)
from tradingview_mcp.core.services.marketaux_service import (
    analyze_sentiment,
    fetch_news_summary,
)
from tradingview_mcp.core.services.yahoo_finance_service import (
    get_price,
    get_price_async,
    get_market_snapshot,
)
from tradingview_mcp.core.services.bitcoin_market_service import get_bitcoin_market_pulse
from tradingview_mcp.core.services.extended_hours_service import (
    get_extended_hours_price,
    get_extended_hours_price_async,
)
from tradingview_mcp.core.services.options_service import (
    get_options_chain,
    get_unusual_options_activity,
)
from tradingview_mcp.core.services.futures_service import (
    get_futures_overview,
    get_futures_movers,
    get_futures_category_snapshot,
    get_futures_watchlist,
)
from tradingview_mcp.core.services.stock_screener_service import (
    EXAMPLE_MARKETS,
    fetch_stock_prices,
    screen_stocks,
)
from tradingview_mcp.core.services.backtest_service import (
    run_backtest,
    compare_strategies as _compare_strategies,
    walk_forward_backtest,
)
from tradingview_mcp.core.utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    normalize_tradingview_symbol,
    normalize_yahoo_symbol,
)
from tradingview_mcp.core.errors import (
    BatchExecutionError,
    ErrorCode,
    make_error,
)

try:
    import tradingview_screener  # noqa: F401
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


# ── Ambil list host yang diizinkan dari env ─────────────────────────────────────
allowed_hosts_env = os.getenv("MCP_ALLOWED_HOSTS", "*")
allowed_hosts = [h.strip() for h in allowed_hosts_env.split(",") if h.strip()]

# ── MCP server instance ────────────────────────────────────────────────────────

mcp = FastMCP(
    name="TradingView Multi-Market Screener",
    instructions=(
        "Multi-market screener backed by TradingView. "
        "Supports crypto exchanges (KuCoin, Binance, Bybit, MEXC, etc.), stock markets "
        "(EGX, BIST, NASDAQ, NYSE, Bursa Malaysia, HKEX, SSE, SZSE, TWSE, TPEX), "
        "and futures markets (CME, COMEX, NYMEX, CBOT — equity index, energy, metals, "
        "agriculture, rates, forex, crypto futures). "
        "Tools: top_gainers, top_losers, bollinger_scan, coin_analysis, multi_agent_analysis, "
        "volume_breakout_scanner, futures_market_overview, futures_top_movers, "
        "futures_category_snapshot, futures_watchlist, egx_market_overview, and more."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=["*"]
    )
)


# ── Screener tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Top Gainers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top gainers for an exchange and timeframe using Bollinger Band analysis.

    Args:
        exchange: Exchange name — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)

    Returns:
        list[dict] on success. On total upstream failure returns a structured
        error envelope: ``{"error": {"code": "ALL_BATCHES_FAILED", ...}}``.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    try:
        rows = await asyncio.to_thread(
            fetch_trending_analysis, exchange, timeframe=timeframe, limit=limit
        )
    except BatchExecutionError as e:
        return make_error(
            ErrorCode.ALL_BATCHES_FAILED, str(e),
            batches_attempted=e.batches_attempted,
            batches_failed=e.batches_failed,
            first_error=e.first_error,
        )
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Top Losers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top losers for an exchange and timeframe. Supports crypto (KUCOIN, BINANCE, MEXC) and stocks (EGX, BIST, NASDAQ).

    Returns ``list[dict]`` on success, or an error envelope on total upstream
    failure (``{"error": {"code": "ALL_BATCHES_FAILED", ...}}``).
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    try:
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    except BatchExecutionError as e:
        return make_error(
            ErrorCode.ALL_BATCHES_FAILED, str(e),
            batches_attempted=e.batches_attempted,
            batches_failed=e.batches_failed,
            first_error=e.first_error,
        )
    rows.sort(key=lambda x: x["changePercent"])
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows[:limit]]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Squeeze Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict]:
    """Scan for assets with low Bollinger Band Width (squeeze detection). Works with crypto and stocks.

    This scans a whole EXCHANGE for squeezes (canonical name is exactly
    `bollinger_scan`; there is no "get_bollinger_band_analysis" tool). For
    the Bollinger read of ONE symbol, call `coin_analysis` instead.

    Example: bollinger_scan(exchange="BINANCE", timeframe="15m", bbw_threshold=0.008)

    Args:
        exchange: Exchange — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M. Typical squeeze thresholds: 15m→0.008, 1h→0.02, 4h→0.04, 1D→0.12
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))
    rows = fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Rating Filter", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "5m", rating: int = 2, limit: int = 25) -> list[dict] | dict:
    """Filter coins by Bollinger Band rating.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, MEXC, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)

    Returns ``list[dict]`` on success, or an error envelope on total upstream
    failure (``{"error": {"code": "ALL_BATCHES_FAILED", ...}}``).
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "5m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    try:
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    except BatchExecutionError as e:
        return make_error(
            ErrorCode.ALL_BATCHES_FAILED, str(e),
            batches_attempted=e.batches_attempted,
            batches_failed=e.batches_failed,
            first_error=e.first_error,
        )
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


# ── Coin / asset analysis ──────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Full Technical Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def coin_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Get detailed analysis for a specific asset (coin or stock) on specified exchange and timeframe.

    Args:
        symbol: Bare ticker, no exchange prefix — crypto: "BTCUSDT", "ETHUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    return analyze_coin(symbol, exchange, timeframe)


# ── Candle pattern tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Consecutive Candles Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20,
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    candle_count = max(2, min(5, candle_count))
    min_growth = max(0.5, min(20.0, min_growth))
    limit = max(1, min(50, limit))
    return scan_consecutive_candles(exchange, timeframe, pattern_type, candle_count, min_growth, limit)


@mcp.tool(annotations=ToolAnnotations(title="Candlestick Pattern Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15,
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    base_timeframe = sanitize_timeframe(base_timeframe, "15m")
    pattern_length = max(2, min(4, pattern_length))
    min_size_increase = max(5.0, min(50.0, min_size_increase))
    limit = max(1, min(30, limit))

    symbols = load_symbols(exchange)
    if not symbols:
        return {"error": f"No symbols found for exchange: {exchange}", "exchange": exchange}
    symbols = symbols[: min(limit * 2, 100)]

    if TRADINGVIEW_SCREENER_AVAILABLE:
        try:
            results = fetch_multi_timeframe_patterns(exchange, symbols, base_timeframe, pattern_length, min_size_increase)
            return {
                "exchange": exchange,
                "base_timeframe": base_timeframe,
                "pattern_length": pattern_length,
                "min_size_increase": min_size_increase,
                "method": "multi-timeframe",
                "total_found": len(results),
                "data": results[:limit],
            }
        except Exception:
            pass

    return scan_advanced_candle_patterns_single_tf(exchange, symbols, base_timeframe, pattern_length, min_size_increase, limit)


# ── Volume scanner tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Volume Breakout Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def volume_breakout_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    volume_multiplier: float = 2.0,
    price_change_min: float = 3.0,
    limit: int = 25,
) -> list[dict] | dict:
    """Detect coins with volume breakout + price breakout."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    volume_multiplier = max(1.5, min(10.0, volume_multiplier))
    price_change_min = max(1.0, min(20.0, price_change_min))
    limit = max(1, min(limit, 50))
    try:
        return await asyncio.to_thread(
            volume_breakout_scan,
            exchange, timeframe, volume_multiplier, price_change_min, limit,
        )
    except BatchExecutionError as e:
        return make_error(
            ErrorCode.ALL_BATCHES_FAILED, str(e),
            batches_attempted=e.batches_attempted,
            batches_failed=e.batches_failed,
            first_error=e.first_error,
        )


@mcp.tool(annotations=ToolAnnotations(title="Volume Confirmation Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def volume_confirmation_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Detailed volume confirmation analysis for a specific coin."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    return volume_confirmation_analyze(symbol, exchange, timeframe)


@mcp.tool(annotations=ToolAnnotations(title="Smart Volume Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def smart_volume_scanner(
    exchange: str = "KUCOIN",
    min_volume_ratio: float = 2.0,
    min_price_change: float = 2.0,
    rsi_range: str = "any",
    limit: int = 20,
) -> list[dict] | dict:
    """Smart volume + technical analysis combination scanner."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
    min_price_change = max(0.5, min(20.0, min_price_change))
    limit = max(1, min(limit, 30))
    try:
        return smart_volume_scan(exchange, min_volume_ratio, min_price_change, rsi_range, limit)
    except BatchExecutionError as e:
        return make_error(
            ErrorCode.ALL_BATCHES_FAILED, str(e),
            batches_attempted=e.batches_attempted,
            batches_failed=e.batches_failed,
            first_error=e.first_error,
        )


# ── Multi-agent analysis ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Agent Market Debate", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def multi_agent_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Run a multi-agent debate (Technical, Sentiment, Risk) for a specific symbol."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    full_symbol = normalize_tradingview_symbol(symbol, exchange)
    return run_multi_agent_analysis(full_symbol, exchange, timeframe)


# ── EGX market tools ───────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="EGX Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_market_overview(timeframe: str = "1D", limit: int = 10) -> dict:
    """Get a comprehensive overview of the Egyptian Exchange (EGX) market."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 20))
    return get_egx_market_overview(timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scan(sector: str = "", timeframe: str = "1D", limit: int = 20) -> dict:
    """Scan EGX stocks by sector. Shows available sectors if none specified."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 50))
    return scan_egx_sector(sector, timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Rotation Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scanner(
    timeframe: str = "1D",
    top_n_sectors: int = 5,
    top_n_stocks: int = 3,
    min_stock_score: int = 60,
) -> dict:
    """Sector rotation scanner for EGX — identifies hot/cold sectors and top picks."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    top_n_sectors = max(1, min(18, top_n_sectors))
    top_n_stocks = max(1, min(10, top_n_stocks))
    min_stock_score = max(0, min(100, min_stock_score))
    return run_egx_sector_scanner(timeframe, top_n_sectors, top_n_stocks, min_stock_score)


@mcp.tool(annotations=ToolAnnotations(title="EGX Index Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_index_analysis(index: str = "EGX30", timeframe: str = "1D", limit: int = 30) -> dict:
    """Analyse an EGX index showing constituent performance with full indicators."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 100))
    return analyze_egx_index(index, timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_stock_screener(
    timeframe: str = "1D",
    min_score: int = 55,
    index_filter: str = "",
    limit: int = 20,
) -> dict:
    """Production stock ranking engine for EGX — finds strong stocks with actionable setups."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    min_score = max(0, min(100, min_score))
    limit = max(1, min(50, limit))
    return screen_egx_stocks(timeframe, min_score, index_filter, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Trade Plan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_trade_plan(symbol: str, timeframe: str = "1D") -> dict:
    """Generate a full trade plan for a specific EGX stock."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    return generate_egx_trade_plan(symbol, timeframe)


@mcp.tool(annotations=ToolAnnotations(title="EGX Fibonacci Retracement", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_fibonacci_retracement(symbol: str, lookback: str = "52W", timeframe: str = "1D") -> dict:
    """Fibonacci retracement analysis for EGX stocks."""
    timeframe = sanitize_timeframe(timeframe, "1D")
    lookback = lookback.strip().upper()
    return analyze_egx_fibonacci(symbol, lookback, timeframe)


# ── Multi-timeframe analysis ───────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Timeframe Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def multi_timeframe_analysis(symbol: str, exchange: str = "KUCOIN") -> dict:
    """Multi-timeframe alignment analysis (Weekly → Daily → 4H → 1H → 15m)."""
    exchange = sanitize_exchange(exchange, "KUCOIN")
    full_symbol = normalize_tradingview_symbol(symbol, exchange)
    return await asyncio.to_thread(run_multi_timeframe_analysis, full_symbol, exchange)


# ── Sentiment & news tools ─────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Market News Sentiment", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_sentiment(symbol: str, category: str = "all", limit: int = 20) -> dict:
    """News sentiment for stocks and crypto (licensed Marketaux entity sentiment)."""
    return analyze_sentiment(symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Financial News Feed", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def financial_news(symbol: str = None, category: str = "stocks", limit: int = 10) -> dict:
    """Real-time financial news via Marketaux (licensed)."""
    return await asyncio.to_thread(fetch_news_summary, symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Combined TA + Sentiment + News", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def combined_analysis(symbol: str, exchange: str = "NASDAQ", timeframe: str = "1D") -> dict:
    """POWER TOOL: TradingView technical analysis + news sentiment + financial news."""
    exchange_clean = sanitize_exchange(exchange, "NASDAQ")
    timeframe_clean = sanitize_timeframe(timeframe, "1D")
    cat = "crypto" if exchange_clean.upper() in ["BINANCE", "KUCOIN", "BYBIT", "MEXC"] else "stocks"

    tech, sentiment, news = await asyncio.gather(
        asyncio.to_thread(analyze_coin, symbol, exchange_clean, timeframe_clean),
        asyncio.to_thread(analyze_sentiment, symbol, cat),
        asyncio.to_thread(fetch_news_summary, symbol, cat, 5),
    )

    tech_momentum = tech.get("market_sentiment", {}).get("momentum", "") if isinstance(tech, dict) else ""
    tech_bullish = tech_momentum == "Bullish"
    sent_bullish = sentiment.get("sentiment_score", 0) > 0.1
    signals_agree = tech_bullish == sent_bullish
    confidence = "HIGH" if signals_agree else "MIXED"
    tech_signal = tech.get("market_sentiment", {}).get("buy_sell_signal", "N/A") if isinstance(tech, dict) else "N/A"

    return {
        "symbol": symbol,
        "exchange": exchange_clean,
        "timeframe": timeframe_clean,
        "technical": tech,
        "sentiment": sentiment,
        "news": {"count": news.get("count", 0), "latest": news.get("items", [])[:3]},
        "confluence": {
            "signals_agree": signals_agree,
            "confidence": confidence,
            "recommendation": (
                f"Technical {tech_signal} "
                f"{'confirmed by' if signals_agree else 'conflicts with'} "
                f"{sentiment.get('sentiment_label', 'Neutral')} news sentiment "
                f"({sentiment.get('posts_analyzed', 0)} articles analyzed)"
            ),
        },
    }


# ── Backtest tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Strategy Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    interval: str = "1d",
    include_trade_log: bool = False,
    include_equity_curve: bool = False,
) -> dict:
    """Backtest a trading strategy on historical data with institutional-grade metrics."""
    return run_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, interval,
        include_trade_log, include_equity_curve,
    )


@mcp.tool(annotations=ToolAnnotations(title="Strategy Comparison Race", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    interval: str = "1d",
) -> dict:
    """Run all 9 strategies and return a ranked leaderboard."""
    return _compare_strategies(symbol, period, initial_capital, interval=interval)


@mcp.tool(annotations=ToolAnnotations(title="Walk-Forward Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def walk_forward_backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "2y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    interval: str = "1d",
) -> dict:
    """Walk-forward backtest to detect overfitting — validates strategy on unseen data."""
    return walk_forward_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, n_splits, train_ratio, interval,
    )


# ── Yahoo Finance tools ────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Real-Time Price Quote", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def yahoo_price(symbol: str) -> dict:
    """Real-time price quote from Yahoo Finance."""
    return await get_price_async(normalize_yahoo_symbol(symbol))


@mcp.tool(annotations=ToolAnnotations(title="Global Market Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_snapshot() -> dict:
    """Global market overview: major indices, top crypto, FX rates, and key ETFs."""
    return get_market_snapshot()


@mcp.tool(annotations=ToolAnnotations(title="Bitcoin Market Pulse", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bitcoin_market_pulse() -> dict:
    """Single-call BTC macro context: price, dominance, total market cap + risk assessment."""
    return get_bitcoin_market_pulse()


@mcp.tool(annotations=ToolAnnotations(title="Extended-Hours Stock Price", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_extended_hours(symbol: str) -> dict:
    """Real-time pre-market and after-hours prices for a US stock symbol."""
    return await get_extended_hours_price_async(symbol)


@mcp.tool(annotations=ToolAnnotations(title="Options Chain", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Full options chain (calls + puts) for a US stock symbol and one expiry."""
    return get_options_chain(symbol, expiry)


@mcp.tool(annotations=ToolAnnotations(title="Unusual Options Activity", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_unusual_activity(
    symbol: str,
    top_n: int = 10,
    min_volume: int = 100,
    expiries: int = 4,
) -> dict:
    """Top strikes by volume / open-interest ratio — institutional positioning signal."""
    return get_unusual_options_activity(symbol, top_n, min_volume, expiries)


# ── Futures tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Futures Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_market_overview(
    category: str = "all",
    exchanges: str = "us",
    limit: int = 30,
    volume_min: int = 0,
) -> dict:
    """Top futures contracts sorted by trading volume."""
    try:
        return get_futures_overview(
            category=category,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        return make_error(ErrorCode.SERVICE_ERROR, f"Futures overview failed: {exc}")


@mcp.tool(annotations=ToolAnnotations(title="Futures Top Movers", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_top_movers(
    direction: str = "gainers",
    exchanges: str = "us",
    limit: int = 20,
    volume_min: int = 10,
) -> dict:
    """Futures contracts with the biggest percentage moves today."""
    direction = direction.lower()
    if direction not in ("gainers", "losers"):
        direction = "gainers"
    try:
        return get_futures_movers(
            direction=direction,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        return make_error(ErrorCode.SERVICE_ERROR, f"Futures movers failed: {exc}")


@mcp.tool(annotations=ToolAnnotations(title="Futures Category Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_category_snapshot(category: str = "energy") -> dict:
    """Quote all major front-month contracts in a specific futures category."""
    return get_futures_category_snapshot(category)


@mcp.tool(annotations=ToolAnnotations(title="Futures Watchlist", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_watchlist() -> dict:
    """Return the full categorized list of well-known front-month futures symbols."""
    return get_futures_watchlist()


@mcp.tool(annotations=ToolAnnotations(title="US Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_screener(
    country: str = "america",
    stock_type: str = "common",
    limit: int = 50,
    exclude_otc: bool = True,
    compact: bool = False,
    sort_by: str = "market_cap",
) -> dict:
    """Screen stocks by share type."""
    try:
        return await asyncio.to_thread(
            screen_stocks, country, stock_type, limit, exclude_otc, compact, sort_by
        )
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"scan failed for market {country!r}: {e}",
            known_good_markets=list(EXAMPLE_MARKETS),
        )


@mcp.tool(annotations=ToolAnnotations(title="Multi-Symbol Stock Prices", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_prices(tickers: str) -> dict:
    """Current price + daily % change for specific stock symbols."""
    try:
        return await asyncio.to_thread(fetch_stock_prices, tickers)
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"price lookup failed: {e}")


# ── Resource ───────────────────────────────────────────────────────────────────

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from the coinlist directory."""
    try:
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "core", "coinlist")
        if not os.path.exists(coinlist_dir):
            return "Coinlist directory not found"
        files = [f for f in os.listdir(coinlist_dir) if f.endswith(".json")]
        return "\n".join([f.replace(".json", "").upper() for f in files])
    except Exception as e:
        return f"Error listing exchanges: {e}"


if __name__ == "__main__":
    # FastMCP uses the standard run() method which reads command line arguments
    # or runs the default streamable HTTP server.
    mcp.run()
