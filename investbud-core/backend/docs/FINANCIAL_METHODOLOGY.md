# Financial Methodology Documentation

This document provides transparent documentation of all financial calculations, formulas, and metrics used in the InvestBud AI backend API. All formulas are presented in mathematical notation to ensure clarity for clients consuming our endpoints.

---

## Table of Contents

1. [Portfolio Return Calculations](#1-portfolio-return-calculations)
2. [Risk Metrics](#2-risk-metrics)
3. [Risk-Adjusted Return Metrics](#3-risk-adjusted-return-metrics)
4. [CAPM & Beta Metrics](#4-capm--beta-metrics)
5. [Growth Metrics](#5-growth-metrics)
6. [Transfer & Transaction Calculations](#6-transfer--transaction-calculations)
7. [Portfolio Composition](#7-portfolio-composition)
8. [Technical Indicators](#8-technical-indicators)
9. [Rolling Window Metrics](#9-rolling-window-metrics)
10. [Assumptions & Parameters](#10-assumptions--parameters)

---

## 1. Portfolio Return Calculations

### 1.1 Simple Returns

The percentage change in price between consecutive periods.

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1
$$

Where:

- $R_t$ = Return at time $t$
- $P_t$ = Price at time $t$
- $P_{t-1}$ = Price at previous period

**Source:** `src/macrocrypto/utils/metrics.py:9-11`

---

### 1.2 Log Returns (Continuously Compounded)

Natural logarithm of price ratios, preferred for statistical analysis.

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})
$$

Where:

- $r_t$ = Log return at time $t$
- $\ln$ = Natural logarithm

**Advantages:**

- Additive across time periods: $r_{total} = \sum_{i=1}^{n} r_i$
- Better statistical properties (approximately normal distribution)
- Handles compounding naturally

**Source:** `src/macrocrypto/utils/metrics.py:14-16`

---

### 1.3 Cumulative Return

Total return over the entire measurement period.

$$
R_{cumulative} = \frac{P_{final}}{P_{initial}} - 1
$$

**Alternative calculation from log returns:**

$$
R_{cumulative} = e^{\sum_{t=1}^{n} r_t} - 1
$$

**Source:** `src/macrocrypto/utils/metrics.py:19-29`

---

### 1.4 Portfolio Weighted Returns

Daily portfolio return weighted by asset composition.

$$
R_{portfolio,t} = \sum_{i=1}^{n} w_{i,t-1} \cdot r_{i,t}
$$

Where:

- $w_{i,t-1}$ = Weight of asset $i$ at previous period (lagged to avoid look-ahead bias)
- $r_{i,t}$ = Log return of asset $i$ at time $t$
- $n$ = Number of assets in portfolio

**Source:** `src/macrocrypto/analytics/wallet_performance.py:54-59`

---

### 1.5 Cumulative Portfolio Returns

Compounded portfolio returns from inception.

$$
R_{cum,t} = e^{\sum_{i=1}^{t} R_{portfolio,i}} - 1
$$

**Source:** `src/macrocrypto/analytics/wallet_performance.py:62`

---

## 2. Risk Metrics

### 2.1 Volatility (Annualized)

Standard deviation of returns, scaled to annual frequency.

$$
\sigma_{annual} = \sigma_{daily} \times \sqrt{365}
$$

Where:

$$
\sigma_{daily} = \sqrt{\frac{1}{n-1} \sum_{t=1}^{n} (R_t - \bar{R})^2}
$$

- $\bar{R}$ = Mean return
- 365 = Trading days per year (crypto markets operate 24/7)

**Source:** `src/macrocrypto/utils/metrics.py:56-73`

---

### 2.2 Maximum Drawdown

The largest peak-to-trough decline in portfolio value.

$$
\text{Drawdown}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}
$$

$$
\text{Max Drawdown} = \min_{t} \left( \text{Drawdown}_t \right)
$$

Where:

- $V_t$ = Portfolio value at time $t$
- $\text{Peak}_t = \max_{s \leq t}(V_s)$ = Running maximum value up to time $t$

**Interpretation:** A max drawdown of -0.30 means the portfolio declined 30% from its peak.

**Source:** `src/macrocrypto/utils/metrics.py:141-155`

---

### 2.3 Value at Risk (VaR)

The maximum expected loss at a given confidence level.

$$
\text{VaR}_{\alpha} = F^{-1}(1 - \alpha)
$$

Where:

- $F^{-1}$ = Inverse of the cumulative distribution function (quantile)
- $\alpha$ = Confidence level (default: 0.95)
- At 95% confidence, VaR is the 5th percentile of returns

**Interpretation:** There is a 95% probability that the portfolio will not lose more than VaR in a single period.

**Source:** `src/macrocrypto/utils/metrics.py:177-188`

---

### 2.4 Conditional Value at Risk (CVaR) / Expected Shortfall

Average loss in the worst-case scenarios beyond VaR.

$$
\text{CVaR}_{\alpha} = \mathbb{E}[R \mid R \leq \text{VaR}_{\alpha}]
$$

**Process:**

1. Calculate VaR at confidence level $\alpha$
2. Average all returns worse than VaR

**Interpretation:** CVaR captures tail risk better than VaR by averaging extreme losses.

**Source:** `src/macrocrypto/utils/metrics.py:191-203`

---

## 3. Risk-Adjusted Return Metrics

### 3.1 Sharpe Ratio

Excess return per unit of total risk.

$$
\text{Sharpe} = \frac{\bar{R} - R_f}{\sigma} \times \sqrt{365}
$$

Where:

- $\bar{R}$ = Mean portfolio return (daily)
- $R_f$ = Risk-free rate (daily)
- $\sigma$ = Standard deviation of portfolio returns
- $\sqrt{365}$ = Annualization factor

**Daily risk-free rate conversion:**

$$
R_{f,daily} = (1 + R_{f,annual})^{\frac{1}{365}} - 1
$$

**Interpretation:**

- Sharpe > 1: Good risk-adjusted return
- Sharpe > 2: Very good
- Sharpe > 3: Excellent

**Source:** `src/macrocrypto/utils/metrics.py:76-103`

---

### 3.2 Sortino Ratio

Excess return per unit of downside risk (penalizes only negative volatility).

$$
\text{Sortino} = \frac{\bar{R} - R_f}{\sigma_{downside}} \times \sqrt{365}
$$

Where the downside deviation is:

$$
\sigma_{downside} = \sqrt{\frac{1}{n_{down}} \sum_{R_t < R_f} (R_t - R_f)^2}
$$

- Only considers returns below the risk-free rate
- Better for asymmetric return distributions

**Interpretation:** Higher Sortino indicates better returns with less downside risk.

**Source:** `src/macrocrypto/utils/metrics.py:106-138`

---

### 3.3 Calmar Ratio

Return per unit of drawdown risk.

$$
\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}
$$

**Interpretation:** Measures how much return is generated relative to the worst historical decline.

**Source:** `src/macrocrypto/utils/metrics.py:158-174`

---

### 3.4 Win Rate

Proportion of periods with positive returns.

$$
\text{Win Rate} = \frac{\text{Count}(R_t > 0)}{n}
$$

Where $n$ = total number of periods.

**Source:** `src/macrocrypto/utils/metrics.py:206-216`

---

## 4. CAPM & Beta Metrics

### 4.1 Beta (Systematic Risk)

Measures portfolio sensitivity to benchmark movements.

$$
\beta = \frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}
$$

**Equivalent regression formulation:**

$$
R_p = \alpha + \beta \cdot R_m + \epsilon
$$

Where:

- $R_p$ = Portfolio returns
- $R_m$ = Benchmark (market) returns
- $\text{Cov}$ = Covariance
- $\text{Var}$ = Variance

**Interpretation:**

- $\beta = 1$: Portfolio moves with the market
- $\beta > 1$: Portfolio is more volatile than market
- $\beta < 1$: Portfolio is less volatile than market
- $\beta < 0$: Portfolio moves opposite to market

**Source:** `src/macrocrypto/utils/advanced_metrics.py:11-62`

---

### 4.2 CAPM Alpha (Jensen's Alpha)

Excess return beyond what CAPM predicts.

$$
\alpha = R_p - \left[ R_f + \beta \cdot (R_m - R_f) \right]
$$

Where:

- $R_p$ = Actual portfolio return
- $R_f$ = Risk-free rate
- $R_m$ = Benchmark return
- $\beta$ = Portfolio beta

**Process:**

1. Calculate expected return: $\text{Expected} = R_f + \beta(R_m - R_f)$
2. Alpha = Actual return - Expected return

**Interpretation:**

- $\alpha > 0$: Portfolio outperformed CAPM prediction (skill/alpha generation)
- $\alpha < 0$: Portfolio underperformed
- $\alpha = 0$: Performance matches CAPM expectation

**Source:** `src/macrocrypto/utils/advanced_metrics.py:112-142`

---

### 4.3 Treynor Ratio

Excess return per unit of systematic (market) risk.

$$
\text{Treynor} = \frac{R_p - R_f}{\beta}
$$

**Difference from Sharpe:** Uses beta (systematic risk) instead of total volatility.

**Interpretation:** Higher Treynor indicates better return for market risk taken.

**Source:** `src/macrocrypto/utils/advanced_metrics.py:82-109`

---

### 4.4 Information Ratio

Active return per unit of tracking error.

$$
\text{IR} = \frac{\bar{R}_p - \bar{R}_b}{\sigma_{tracking}} \times \sqrt{365}
$$

Where:

$$
\sigma_{tracking} = \text{StdDev}(R_p - R_b)
$$

- Active return = Portfolio return - Benchmark return
- Tracking error = Standard deviation of active returns

**Interpretation:** Measures skill in generating returns different from benchmark.

**Source:** `src/macrocrypto/analytics/wallet_performance.py:241-269`

---

### 4.5 Excess Return

Simple difference between portfolio and benchmark returns.

$$
\text{Excess Return} = R_p - R_b
$$

**Source:** `src/macrocrypto/utils/advanced_metrics.py:65-79`

---

## 5. Growth Metrics

### 5.1 CAGR (Compound Annual Growth Rate)

Annualized growth rate assuming smooth compounding.

$$
\text{CAGR} = \left( \frac{V_{final}}{V_{initial}} \right)^{\frac{1}{years}} - 1
$$

Where:

$$
years = \frac{\text{days}}{365.25}
$$

**Example:** Initial = \$1,000, Final = \$2,000, Period = 5 years

$$
\text{CAGR} = \left( \frac{2000}{1000} \right)^{\frac{1}{5}} - 1 = 14.87\%
$$

**Source:** `src/macrocrypto/utils/metrics.py:32-53`

---

## 6. Transfer & Transaction Calculations

### 6.1 Gas Cost Calculation

For Ethereum and EVM-compatible chains:

$$
\text{Gas Cost (ETH)} = \frac{\text{Gas Used} \times \text{Effective Gas Price}}{10^{18}}
$$

Where:

- Gas Used = Units of gas consumed by transaction
- Effective Gas Price = Wei per gas unit
- $10^{18}$ = Wei to ETH conversion (1 ETH = $10^{18}$ Wei)

**Note:** Gas costs are only counted when the wallet is the transaction sender (paid gas).

**Source:** `src/macrocrypto/services/wallet_history_service.py:244-308`

---

### 6.2 Token Balance from Transfers

Running balance calculated from cumulative transfers.

$$
\text{Balance}_{t} = \sum_{i=1}^{t} \text{Transfer Value}_i
$$

For inflows (receiving): positive value
For outflows (sending): negative value

**Source:** `src/macrocrypto/services/wallet_history_service.py:450-456`

---

### 6.3 ETH Balance Reconciliation

For ETH specifically, we use on-chain balances to account for gas:

$$
\text{Gas Spent} = \text{Balance}_{calculated} - \text{Balance}_{on-chain}
$$

Where:

- $\text{Balance}_{calculated}$ = Sum of transfers
- $\text{Balance}_{on-chain}$ = Actual balance from `eth_getBalance`

**Source:** `src/macrocrypto/services/wallet_history_service.py:458-512`

---

### 6.4 Token Decimal Normalization

Converting from smallest units to human-readable format:

$$
\text{Balance}_{normalized} = \frac{\text{Balance}_{raw}}{10^{decimals}}
$$

**Examples:**

- ETH: $\frac{1,000,000,000,000,000,000}{10^{18}} = 1$ ETH
- USDC: $\frac{1,000,000}{10^{6}} = 1$ USDC

**Source:** `src/macrocrypto/services/wallet_history_service.py:236-243`

---

### 6.5 Portfolio Value

Total USD value across all holdings.

$$
V_{portfolio,t} = \sum_{i=1}^{n} \text{Balance}_{i,t} \times \text{Price}_{i,t}
$$

Where:

- $\text{Balance}_{i,t}$ = Token quantity (forward-filled if no transfer)
- $\text{Price}_{i,t}$ = USD price on date $t$
- $n$ = Number of tokens held

**Source:** `src/macrocrypto/services/wallet_history_service.py:546-547`

---

## 7. Portfolio Composition

### 7.1 Asset Weights

Proportion of portfolio value in each asset.

$$
w_{i,t} = \frac{V_{i,t}}{V_{portfolio,t}}
$$

Where:

- $V_{i,t}$ = USD value of asset $i$ at time $t$
- $\sum_{i=1}^{n} w_{i,t} = 1$ (weights sum to 100%)

**Source:** `src/macrocrypto/analytics/wallet_performance.py:51-52`

---

### 7.2 Risk vs Stable Allocation

$$
\text{Risk Allocation} = \frac{\sum_{i \in \text{risk}} V_i}{V_{portfolio}}
$$

$$
\text{Stable Allocation} = \frac{\sum_{i \in \text{stable}} V_i}{V_{portfolio}}
$$

**Stablecoins include:** USDC, USDT, DAI, BUSD, FRAX, and dynamically fetched from CoinGecko.

**Source:** `src/macrocrypto/data/wallet_analyzer.py:418-469`

---

## 8. Technical Indicators

### 8.1 RSI (Relative Strength Index)

Momentum oscillator measuring speed and magnitude of price changes.

$$
\text{RSI} = 100 - \frac{100}{1 + RS}
$$

Where:

$$
RS = \frac{\text{Average Gain}}{\text{Average Loss}}
$$

Over a 14-day lookback period (default).

**Interpretation:**

- RSI > 70: Overbought (potential pullback)
- RSI < 30: Oversold (potential bounce)
- RSI = 50: Neutral

**Source:** `src/macrocrypto/data/btc_data.py:89-121`

---

### 8.2 Moving Averages

Simple moving averages over various windows:

$$
\text{MA}_n = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}
$$

**Calculated for:** 7, 30, 90, 200-day windows.

**Price relative to MA:**

$$
\text{Price vs MA} = \frac{P_t - \text{MA}_t}{\text{MA}_t}
$$

**Golden Cross:** $\text{MA}_{30} > \text{MA}_{200}$ (bullish)
**Death Cross:** $\text{MA}_{30} < \text{MA}_{200}$ (bearish)

**Source:** `src/macrocrypto/data/btc_data.py:192-203`

---

### 8.3 Drawdown from All-Time High

$$
\text{Drawdown}_{ATH,t} = \frac{P_t - \text{ATH}_t}{\text{ATH}_t}
$$

Where:

$$
\text{ATH}_t = \max_{s \leq t}(P_s)
$$

**Source:** `src/macrocrypto/data/btc_data.py:123-149`

---

### 8.4 Rolling Volatility

$$
\sigma_{rolling,t} = \text{StdDev}(R_{t-n+1}, ..., R_t) \times \sqrt{365}
$$

**Calculated for:** 7, 30, 90-day rolling windows.

**Source:** `src/macrocrypto/data/btc_data.py:151-173`

---

## 9. Rolling Window Metrics

All rolling metrics use a 30-day window by default.

### 9.1 Rolling Sharpe Ratio

$$
\text{Sharpe}_{rolling,t} = \frac{\bar{R}_{[t-30:t]} - R_f}{\sigma_{[t-30:t]}} \times \sqrt{365}
$$

**Source:** `src/macrocrypto/analytics/wallet_performance.py:356-368`

---

### 9.2 Rolling Sortino Ratio

$$
\text{Sortino}_{rolling,t} = \frac{\bar{R}_{[t-30:t]} - R_f}{\sigma_{downside,[t-30:t]}} \times \sqrt{365}
$$

**Source:** `src/macrocrypto/analytics/wallet_performance.py:370-393`

---

### 9.3 Rolling Volatility

$$
\sigma_{rolling,t} = \text{StdDev}(R_{[t-30:t]}) \times \sqrt{365}
$$

**Source:** `src/macrocrypto/analytics/wallet_performance.py:395-403`

---

### 9.4 Rolling Beta

$$
\beta_{rolling,t} = \frac{\text{Cov}(R_{p,[t-30:t]}, R_{m,[t-30:t]})}{\text{Var}(R_{m,[t-30:t]})}
$$

**Source:** `src/macrocrypto/analytics/wallet_performance.py:405-452`

---

## 10. Assumptions & Parameters

### Global Constants

| Parameter         | Value     | Rationale                       |
| ----------------- | --------- | ------------------------------- |
| Trading days/year | 365       | Crypto markets operate 24/7     |
| Risk-free rate    | 2% annual | Default; can use Fed Funds rate |
| VaR confidence    | 95%       | Industry standard               |
| RSI period        | 14 days   | Standard technical analysis     |
| Rolling window    | 30 days   | Monthly evaluation period       |

### Fee Assumptions (Backtesting)

| Fee Type        | Default Value | Description               |
| --------------- | ------------- | ------------------------- |
| Slippage        | 1%            | Market impact of trade    |
| Pool/Swap fee   | 0.3%          | Typical DEX fee (Uniswap) |
| Transaction fee | $5 USD        | Network gas cost          |

### Data Handling

| Scenario          | Treatment                                   |
| ----------------- | ------------------------------------------- |
| Missing prices    | Forward-filled from last known              |
| Missing balances  | Forward-filled (no transfer = same balance) |
| NaN returns       | Dropped from calculations                   |
| Infinite values   | Replaced with NaN                           |
| First day returns | Set to 0 (no prior period)                  |
| Zero division     | Returns 0 or NaN with appropriate handling  |

### Benchmark

- **Default benchmark:** Bitcoin (BTC)
- Used for beta, alpha, and relative performance calculations

---

## Quick Reference Table

| Metric            | Formula                                  | Higher is Better?   |
| ----------------- | ---------------------------------------- | ------------------- |
| Simple Return     | $(P_t / P_{t-1}) - 1$                    | Yes                 |
| Volatility        | $\sigma \times \sqrt{365}$               | No (risk measure)   |
| Sharpe Ratio      | $(\bar{R} - R_f) / \sigma$               | Yes                 |
| Sortino Ratio     | $(\bar{R} - R_f) / \sigma_{down}$        | Yes                 |
| Max Drawdown      | $\min((V - Peak)/Peak)$                  | No (closer to 0)    |
| Beta              | $\text{Cov}(R_p, R_m) / \text{Var}(R_m)$ | Depends on strategy |
| Alpha             | $R_p - [R_f + \beta(R_m - R_f)]$         | Yes                 |
| CAGR              | $(V_f / V_i)^{1/years} - 1$              | Yes                 |
| VaR 95%           | 5th percentile of returns                | No (closer to 0)    |
| CVaR 95%          | Mean of returns $\leq$ VaR               | No (closer to 0)    |
| Win Rate          | Positive days / Total days               | Yes                 |
| Calmar            | CAGR / \|Max DD\|                        | Yes                 |
| Treynor           | $(R_p - R_f) / \beta$                    | Yes                 |
| Information Ratio | Active return / Tracking error           | Yes                 |

---

_Last updated: November 2025_
_For questions about methodology, contact the InvestBud AI team._
