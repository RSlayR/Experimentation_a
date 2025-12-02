# Experimentation_a

This repository contains a CFA-inspired portfolio optimizer that balances ETF-heavy allocations with configurable risk profiles. The optimizer leans on CAPM inputs, mean-variance objectives, and ridge/lasso-style penalties to keep allocations sparse and ETF-focused.

## Features
- **Risk-profile aware optimization:** choose conservative, balanced, or growth settings with distinct risk aversion, regularization strengths, and per-position caps.
- **ETF preference:** penalizes allocations to stocks so ETFs are favored when both are available.
- **Regularization:** L1 (lasso) and L2 (ridge) terms temper concentration and limit the number of active tickers.
- **CAPM estimates:** derives expected returns from betas against a market proxy and a configurable risk-free rate.
- **Performance diagnostics:** computes Sharpe, Treynor, Jensen's alpha, and M² while plotting the portfolio against the risk-free point and market portfolio.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end demo (downloads 3 years of data for sample ETFs, stocks, and a market proxy):
   ```bash
   python cfa_portfolio.py
   ```
3. Customize tickers, risk-free rate, and profiles by editing the `demo()` function or importing the helper functions in your own script.

## Key Files
- `cfa_portfolio.py`: Optimizer implementation, plotting utilities, and demo entrypoint.
- `requirements.txt`: Minimal dependencies for running the optimizer and plots.
