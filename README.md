# Sablier MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![MCP 1.26+](https://img.shields.io/badge/MCP-1.26+-green.svg)](https://modelcontextprotocol.io)

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that lets AI assistants analyze portfolios, stress-test scenarios, generate synthetic market paths, and scan SEC filings — in under 2 minutes.

## What This Does

Connect Sablier to Claude, ChatGPT, or any MCP-compatible AI assistant. The agent gets 53 tools to:

**Scan SEC filings & earnings calls** — AI reads every company's 10-K, 10-Q filings and earnings call transcripts, then scores how exposed each holding is to any theme you ask about (0-100 scale with evidence). _"How exposed is my portfolio to China supply chain risk?"_ — scored, evidenced, and ranked in seconds.

**Compute factor exposures** — Measures how each stock responds to market drivers (interest rates, VIX, dollar index, oil, credit spreads, etc.). Factor betas are estimated on a rolling window of recent data so they reflect current market conditions. Supports both linear and nonlinear (GAM) factor models.

**Stress-test with scenarios** — _"What if VIX hits 40?"_ or _"What if the Fed raises rates to 6%?"_ Run Monte Carlo simulations to get per-asset expected returns, Value-at-Risk, Expected Shortfall, and full return distributions.

**Generate synthetic market paths** — Train a generative flow model on the joint distribution of assets and factors. Generate hundreds of realistic future trajectories, optionally constrained (_"paths where gold stays above $3000"_). Forward-test strategies across many scenarios.

**Real-time market intelligence** — Get a Bloomberg-terminal-grade briefing with 50+ indicators, z-scores, regime signals, and cross-asset analysis in one call.

**Manage portfolios** — Create and track portfolios with live prices, performance analytics (Sharpe ratio, max drawdown, volatility), optimization, and efficient frontier computation.

### Speed

> **Under 2 minutes, end-to-end.** Portfolio creation -> model training -> factor betas -> stress scenarios -> SEC filing analysis. All in a single conversation.
>
> The same workflow — gathering filings, building factor models, running simulations, writing risk memos — takes a team of analysts and quants **days to weeks**. Sablier compresses it into one chat.

## Quick Start

### Option A: Claude Desktop / Claude.ai (recommended — zero install)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sablier": {
      "type": "url",
      "url": "https://sablier-mcp-215397666394.us-central1.run.app/mcp/"
    }
  }
}
```

That's it. On first use, Claude opens a browser window — sign up or log in with your Sablier account. No API keys needed.

### Option B: ChatGPT (Developer Mode)

1. Go to **Settings -> Developer Mode** and enable it
2. Go to **Connectors -> Add connector**
3. Enter:
   - **Name**: `Sablier`
   - **Server URL**: `https://sablier-mcp-215397666394.us-central1.run.app/mcp/`
4. In a new chat, activate via **"+" -> More -> Developer Mode**

### Option C: Claude Code (local, stdio)

```bash
# Clone and install
git clone https://github.com/miradebs98/sablier-mcp.git
cd sablier-mcp
uv sync

# Register with Claude Code
claude mcp add sablier -- uv --directory /path/to/sablier-mcp run sablier-mcp

# Set your API key (get one from sablier-ai.com)
export SABLIER_API_KEY=sk_live_your_key_here
```

## Tools (53)

### Market Intelligence

| Tool | Description |
|------|-------------|
| `market_radar` | Bloomberg-grade briefing: 50+ indicators, z-scores, regime signals, cross-asset analysis |
| `search_features` | Search for tickers (stocks, ETFs, futures) and market indicators (VIX, DXY, rates) |

### Portfolio Management

| Tool | Description |
|------|-------------|
| `create_portfolio` | Create a portfolio from tickers and weights (must sum to 1.0) |
| `list_portfolios` | List your portfolios with names, assets, and status |
| `get_portfolio` | Get full details for a specific portfolio |
| `update_portfolio` | Update name, description, weights, or capital |
| `delete_portfolio` | Permanently delete a portfolio |
| `get_portfolio_value` | Live portfolio value: total, P&L, per-position breakdown |
| `get_portfolio_analytics` | Sharpe ratio, volatility, max drawdown, beta (1W-5Y timeframes) |
| `get_asset_profiles` | Sector, industry, country, and exchange for each holding |
| `optimize_portfolio` | Find optimal weights: max Sharpe, min variance, or max return |
| `get_efficient_frontier` | Compute the efficient frontier curve for portfolio assets |

### Qualitative Analysis (SEC Filings & Earnings Calls)

| Tool | Description |
|------|-------------|
| `analyze_qualitative` | Score company exposure to any theme (0-100) using 10-K, 10-Q, and earnings transcripts |
| `list_themes` | Browse the built-in theme library (AI risk, rate sensitivity, China exposure, etc.) |
| `list_grain_analyses` | List past qualitative analyses |
| `get_grain_analysis` | Load a saved analysis with full scores and evidence passages |
| `delete_grain_analysis` | Delete a saved qualitative analysis |

### Quantitative Analysis (Factor Models)

| Tool | Description |
|------|-------------|
| `analyze_quantitative` | One-shot: builds factor models, trains, computes factor exposures (linear + nonlinear) |
| `list_model_groups` | List existing analyses with training and simulation status |
| `list_feature_set_templates` | Browse pre-built market driver sets (rates, volatility, commodities, credit, etc.) |
| `create_feature_set` | Create a custom set of market drivers for analysis |
| `list_feature_sets` | List all accessible feature sets |
| `get_feature_set` | Get details of a specific feature set |
| `delete_feature_set` | Delete a custom feature set |
| `simulate_betas` | Compute per-asset factor betas from a trained model group |
| `run_model_validation` | Validate model quality: R-squared, autocorrelation, regime sensitivity |
| `get_model_validation` | Get cached validation results |
| `get_residual_correlation` | Cross-asset residual correlation matrix |
| `list_simulations` | List all simulations for a model group |
| `delete_model_group` | Delete a model group and all associated data |

### Stress Testing & Scenarios

| Tool | Description |
|------|-------------|
| `simulate_returns` | Monte Carlo what-if: per-asset VaR, ES, expected return under custom factor levels |
| `run_scenario` | Run a saved scenario template |
| `create_scenario` | Save a named what-if scenario (fixed value, percentile, or shock) |
| `list_scenarios` | List saved scenarios |
| `get_scenario` | Get scenario details |
| `update_scenario` | Update a scenario's factors or description |
| `delete_scenario` | Delete a saved scenario |
| `clone_scenario` | Clone a scenario template for editing |

### Generative Simulation (Flow)

| Tool | Description |
|------|-------------|
| `generate_synthetic` | One-shot: train a flow model and generate multi-step synthetic market paths |
| `simulate_flow_scenario` | Generate constrained paths (e.g., "gold above $3000 and VIX below 20") |
| `test_flow_risk` | Compute portfolio risk metrics across generated paths |
| `flow_validate` | Validate flow model quality against historical data |

### Feature Catalog

| Tool | Description |
|------|-------------|
| `add_feature` | Add a ticker to the catalog (Yahoo Finance or FRED) |
| `refresh_feature_data` | Fetch/update historical data for tickers |
| `create_derived_feature` | Create derived features (moving averages, spreads, ratios) |
| `list_transformations` | List available transformation types |

### API Keys & Billing

| Tool | Description |
|------|-------------|
| `set_api_key` | Store a third-party API key (FRED, Finnhub) |
| `list_api_keys` | List stored API key providers |
| `delete_api_key` | Remove a stored API key |
| `get_billing_info` | View current subscription tier and limits |
| `get_billing_usage` | View usage across metered buckets |
| `subscribe` | Subscribe or upgrade via Stripe Checkout |
| `manage_subscription` | Open Stripe Customer Portal |

## Example Conversations

### 1. Full risk analysis in one conversation

```
You:   Create a portfolio with AAPL 40%, MSFT 30%, NVDA 30%.
       Then stress-test it for a recession — VIX at 35, 10Y at 5.5%, SPY at 380.

Agent: 1. create_portfolio("Tech Portfolio", ["AAPL", "MSFT", "NVDA"], [0.4, 0.3, 0.3])
       2. list_feature_set_templates()  →  picks "Macro + Volatility" set
       3. analyze_quantitative(portfolio_id, conditioning_set_id)
          →  trains models, computes factor betas per asset
       4. simulate_returns(sim_batch_id, {"VIX": 35, "US 10Y": 5.5, "SPY": 380})
          →  per-asset expected returns, VaR (95%), Expected Shortfall

       Result: Portfolio expected return = -8.2%, VaR(95%) = -14.5%
       NVDA most exposed (-12.1%), MSFT most defensive (-4.8%)
```

### 2. SEC filing analysis for thematic risk

```
You:   How exposed are the Magnificent 7 to AI regulation risk?

Agent: 1. analyze_qualitative(
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
            themes=["AI regulation risk"]
       )
       2. Returns per-ticker scores (0-100) with evidence from 10-K filings:
          - META: 82/100 (HIGH) — "government regulation of AI... could limit our ability
            to deploy AI features across our family of apps"
          - GOOGL: 78/100 — "new AI regulations could require us to restrict or modify
            products and services"
          - NVDA: 71/100 — "export controls... restrictions on AI chip sales to China"
          - TSLA: 31/100 (LOW) — limited AI regulation mentions in filings
```

### 3. Generative scenario analysis

```
You:   I want to see how a gold + bonds portfolio performs over the next quarter
       in scenarios where inflation stays high.

Agent: 1. create_portfolio("Inflation Hedge", ["GLD", "TLT", "IAU"], [0.4, 0.4, 0.2])
       2. generate_synthetic(portfolio_id, conditioning_set_id, horizon=60, n_paths=500)
          →  trains flow model (~5 min), generates 500 joint price paths
       3. simulate_flow_scenario(model_group_id, constraints=[
            {"feature_name": "CPI", "type": "level", "lower": 3.5, "t_start": 0, "t_end": 60}
          ])
          →  generates paths conditioned on CPI > 3.5%
       4. test_flow_risk(portfolio_id, job_id)
          →  distribution of Sharpe, max drawdown, total return across constrained paths

       Result: Median return = +4.2%, 5th percentile = -6.8%
       Gold outperforms bonds in 72% of high-inflation paths
```

### 4. Market briefing and portfolio checkup

```
You:   What's happening in markets today and how is my portfolio positioned?

Agent: 1. market_radar()
          →  50+ indicators: equities, rates, credit, FX, commodities, volatility
          →  regime signals: Risk-Off score = 0.7, yield curve inverted, VIX elevated
       2. list_portfolios()  →  finds your "Tech Portfolio"
       3. get_portfolio_value(portfolio_id)
          →  current value, daily P&L, per-position breakdown
       4. get_portfolio_analytics(portfolio_id, timeframe="1M")
          →  1-month Sharpe, volatility, max drawdown

       Result: Markets are risk-off (VIX +15% this week, credit spreads widening).
       Your tech portfolio is down -2.3% today, concentrated in high-beta names.
       Consider: stress-test with simulate_returns to quantify downside risk.
```

## Authentication

- **Remote mode** (Claude Desktop, Claude.ai, ChatGPT): OAuth 2.0 browser-based login — no API keys to manage. Sablier authenticates via Google OAuth, then issues an API key for the MCP session.
- **Local mode** (Claude Code, stdio): Set `SABLIER_API_KEY` environment variable.

## Pricing

| Tier | Price | Included |
|------|-------|----------|
| Free | $0 | 10 market radar, 5 factor models, 2 GRAIN analyses/mo |
| Pro | $79/mo | 100 market radar, 50 factor models, 20 GRAIN, 10 Flow sims/mo |
| Enterprise | $399/mo/seat | Unlimited everything, priority support |

Portfolio management, read operations, and scenario management are always free.
Overages billed per-call beyond included limits.

## Architecture

```
sablier-mcp/
├── src/sablier_mcp/
│   ├── server.py      # 53 MCP tool definitions (FastMCP)
│   ├── client.py      # Async HTTP client for Sablier API
│   ├── auth.py        # OAuth 2.0 provider (remote mode)
│   └── widgets.py     # Rich HTML cards for Claude Desktop
├── pyproject.toml
├── Dockerfile
└── README.md
```

- **Remote mode** (Claude Desktop, Claude.ai, ChatGPT): OAuth 2.0 browser login — no API keys to manage
- **Local mode** (Claude Code, stdio): API key from environment variable
- **Widgets**: Tools return rich HTML cards (beta heatmaps, score cards, portfolio overviews) alongside text for visual output in Claude Desktop

## Development

```bash
# Run the server locally (stdio transport)
uv run sablier-mcp

# Test with MCP inspector
npx @modelcontextprotocol/inspector uv --directory . run sablier-mcp

# Run as remote server (streamable-http with OAuth)
MCP_TRANSPORT=streamable-http uv run sablier-mcp
```

## Privacy Policy

Sablier processes portfolio data and market queries to provide analytics. Full details:

- **Data collected**: Portfolio holdings, factor model parameters, and query metadata for analytics computation
- **Usage & storage**: Data is processed on Sablier's servers (GCP, US) and stored for your account's analytics history. Portfolios and models persist until you delete them.
- **Third-party sharing**: Sablier does not sell or share your data. Market data is sourced from public feeds. SEC filings are public records.
- **Retention**: Account data is retained while your account is active. Deleted portfolios and models are purged within 30 days.
- **Contact**: [team@sablier.it](mailto:team@sablier.it)

Full privacy policy: [sablier-ai.com/privacy](https://sablier-ai.com/privacy) | Terms of service: [sablier-ai.com/terms](https://sablier-ai.com/terms)

## Support

- **Email**: [team@sablier.it](mailto:team@sablier.it)
- **Issues**: [github.com/miradebs98/sablier-mcp/issues](https://github.com/miradebs98/sablier-mcp/issues)

## Links

- **Sablier Platform**: [sablier-ai.com](https://sablier-ai.com/discover)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
