# Algorithmic Trading: Statistical Arbitrage (Pairs Trading)
Main source: "Quantitative Trading: How to Build Your Own Algorithmic Trading Business" by Ernie Chan

Other sources: 

[Arbitrage Strategies: Understanding Working of Statistical Arbitrage](https://blog.quantinsti.com/statistical-arbitrage/)

[Pairs Trading for Beginners: Correlation, Cointegration, Examples, and Strategy Steps](https://blog.quantinsti.com/pairs-trading-basics/)

[Augmented Dickey Fuller (ADF) Test for a Pairs Trading Strategy](https://blog.quantinsti.com/augmented-dickey-fuller-adf-test-for-a-pairs-trading-strategy/)



## Terminology, Concepts

### Greeks
- **Alpha**: excess return of a strategy or investment relative to a benchmark (e.g. S&P 500); positive means outperform, negative means underperform
- **Beta**: systematic risk i.e. how much a security moves relative to overall market; =1 moves in-sync with market, >1 more volatile, <1 less volative

### Volatlity

- Measure of how much the price of an asset fluctuates over time; high volatility = large, unpredictable swings (e.g. meme stocks); low volatility = steady, slow-moving prices (e.g., treasury bonds)
- Volatility = risk; it affects option pricing, portfolio construction, risk management, and execution strategies
- Volatility = SD of returns $\sqrt{E(R - \mu)^2}$, where R = asset’s return over some time period, mu = average return, E = expectation (mean)
- Daily returns → for short-term volatility, log returns → for compounding behavior, annualized volatility → scaled to yearly by multiplying by $\sqrt{252}$ (trading days)


### Risk
- Risk = uncertainty. high variance = returns fluctuate a lot = unpredictable outcomes, low variance = returns are stable and more predictable
- Investors don’t just care about average return — they care about how volatile those returns are; even if two portfolios have the same average return, the one with higher variance is riskier.

| Day | Strategy A Return | Strategy B Return |
|-----|-------------------|-------------------|
| 1   | 1%                | -5%               |
| 2   | 1%                | 10%               |
| 3   | 1%                | -3%               |
| 4   | 1%                | 8%                |
| 5   | 1%                | -6%               |

Strategy A has zero variance — returns are constant. Strategy B has high variance — unpredictable outcomes, higher risk.


### Risk-free Rate
- Theoretical return from an investment with zero risk of loss, meaning no chance of default and no uncertainty about future cash flows
- In reality, there’s no such thing as truly risk-free, but U.S. government’s short-term Treasury bills (T-bills) are used as a proxy, because unlikely to default, are highly liquid, short-term (typically 1–12 months), and have minimal interest rate risk


### Neutral Portfolios
- Investment strategy designed to eliminate or minimize exposure to a specific market risk — such as movements in the overall stock market — while still allowing the investor to profit from relative performance between assets

| Type              | Neutral To                 | Description / Example                                                                 |
|-------------------|-----------------------------|----------------------------------------------------------------------------------------|
| **Market Neutral**| Overall market direction    | Long undervalued stocks and short overvalued ones to eliminate exposure to market moves |
| **Sector Neutral**| Specific sectors or industries | Equal exposure across sectors to avoid bias toward any one industry                   |
| **Dollar Neutral**| Dollar exposure (long = short) | Long $100K in one stock, short $100K in another                                       |
| **Beta Neutral**  | Market beta (target beta = 0) | Adjust positions so overall sensitivity to market is zero (e.g., using CAPM beta)     |



## Backtesting, Performance Measures
- The process of testing a trading strategy or model on historical data to see how it would have performed in the past
- Use open/close data vs. high/low data (less noisy) and adjusted for splits and dividends. If a company issues a dividend $\$d$ per share, all prices before $T$ ned to be multiplied by $(\text{Close}(T - 1) - d) / \text{Close}(T - 1)$ where $Close$ is the closing price of a specific day and $T - 1$ is the trading day before $T$.
- Subtract risk-free rate from returns of dollar-neutral portolio or no?
- "The answer is no. A dollar-neutral portfolio is self-financing, meaning the cash you get from selling short pays for the purchase of the long securities, so the financing cost (due to the spread between the credit and debit interest rates) is small and can be neglected for many backtesting purposes. Meanwhile, the margin balance you have to maintain earns a credit interest close to the risk-free rate $r_F$. So let’s say the strategy return (the portfolio return minus the contribution from the credit interest) is $R$, and the risk-
free rate is $r_F$. Then the excess return used in calculating the Sharpe ratio is $R + rF– rF = R$. So, essentially, you can ignore the risk-free rate in the whole calculation and just focus on the returns due to your stock positions."
- Subtract risk-free rate when calculating Sharpe ratio iff financing cost

### Sharpe Ratio
$\text{Sharpe Ratio} = \frac{\text{Average of Excess Returns}}{\text{SD of Excess Returns}} \: \text{where Excess Returns} = \text{Portfolio Returns} − \text{Benchmark Returns}$
<br> <br>
Equivalently, $\frac{E(R_p - R_f)}{\sigma_p}$ <br> <br>
where $R_p$ = portfolio/strategy return <br>
$R_f$ = risk-free rate (often T-bill rate or 0) <br>
$\sigma_p$ = SD of returns (volatility) <br>
$E(\cdot)$ = expected/average value

- Tells you how much excess return you’re getting per unit of risk (volatility). It answers the question: “Am I being rewarded for the risk I’m taking?”
- If a strategy trades infrequently, the Sharpe Ratio is probably not high, converse of a strategy that has long or high drawdowns
- <1 ratio is not suitable standalone, annualized ratio for monthly profitability is >2, >3 for daily profitability
- Subtracted risk-free rate usually


### Drawdown
- Strategy suffers drawdown when it loses money recently: at time $t$, difference in current equity and global maximum before $t$
- Maximum drawdown: difference between global max equity (high watermark) and global min equity, given that the min occurred after the max
- Max drawdown duration: longest it has taken for the equity curve to recover losses



## Statistical Arbitrage (Strategy)


### Arbitrage
- Simultaneously transacting multiple financial securities for profit due to a difference in prices


### Types of StatArb
- Market Neutral: exploit increasing/decreasing prices in one or more markets
- Cross Market: exploit price discrepancy of same asset across different markets i.e. buy asset in cheaper market and sells in more expensive market
- Cross Asset: exploit price discrepancy between financial assets and its underlying
- ETF: exploit price discrepancies between value of ETF and underlying assets; type of cross asset arbitrage

### Mean Reversion
- Deviation of price or spread from mean will revert back to mean eventually


### Pair Trading
- Hedging two stocks against each other (one moving up, one moving down), traded in a market-neutral strategy


### Cointegration
- Spread = $log(a) - n log(b)$ where a, b = prices of stocks A, B, respectively; for each stock of A bought, sell n stocks of B
- If A and B are cointegrated, spread is stationary (mean and variance of equation is constant over time)
- Stationarity: statistical properties of time series (e.g. mean, variance, covariance) do not change over time (has no unit root)


### Hedge Ratio
- $n$ is the hedge ratio; $n = 0$ implies that EV of spread will remain 0 --> any deviation is statistically abnormal --> grounds for pair trading
- Assume constant for pairs trading; calculated by regressing prices of A and B


### Augmented Dickey-Fuller (ADF) Test
- Test for likeliness of cointegration based on fluctuation of residuals of OLS predictions 
- Hypotheses:
    - Null: there exists a unit root and the time-series is not stationary
    - Alternate: there is no unit root and the time-series is stationary
- If the spread (residuals) just fluctuates randomly around a mean (typically around 0), it is stationary: the two stocks move together in the long run.
- If the spread drifts or trends, it’s non-stationary: no stable relationship, so the pair is not reliable
- $H_0$: spread has a unit root (non-stationary)
- $H_1$: spread is stationary (mean-reverting)
- If p-value < 0.05, reject $H_0$ --> spread is stationary --> good candidate for pairs trading


### Entry Points
- Calculate z-score of spread for a time period
- Define threshold between two different SD points (e.g. 1.5 sigma - 2 sigma)
- When $lower < z < upper$, go SHORT: sell A, buy B
- When $-upper < z < -lower$, go LONG: buy A, sell B
- Maintain hedge ratio to calculate stock quanitty


### Exit Points
- Stop Loss: expected outcome does not occur e.g. spread exceeds upper threshold and doesn't revert to mean. Example: threshold his 1.5 - 2-sigma, spread hits 2.5-sigma; place stop loss at 3-sigma
- Take profit: take profit before prices move in other direction. If spread moves as expected and reverts back to mean, exit trades when close to mean or when just crossed. 

## Biases


### Survivorship Bias
- Historical database that only include stocks that have "survived" bankruptcies, delistings, mergers, or acquisitions
- Can be dangerous when backtesting and cause to misleading results because value might be skewed



### Look-Ahead Bias
- Using information/data that was only available in the future of the instant the trade was executed
- To avoid, use lagged historical data for calculating signals, only using data from previous trading day only



### Data-Snooping Bias
- Overfitting model with lots of parameters to historial accidents may fail b/c these events probably won't repeat themselves
- Simple models are less prone to suffering from data-snooping bias
- Most basic safeguard against data-snooping bias is to ensure sufficient amount of backtest data relative to number of free parameters to optimize; use training and testing sets 
- Sensitivity analysis: once parameters have been optimized and features have been verified to perform relatively well on test set, alter parameters slightly to see changes in training and testing sets; if drastic changes, likely suffer data-snooping bias


### Strategy Refinement
- Changes made to improve the training set must also improve the testing set
- Exclude certain industry sectors (extra sensitive to politics/news), pending acquisitions/merger deals; consider cap size 
