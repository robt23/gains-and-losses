# Options Trading

Sources:
- [Options Trading Guide, Investopedia](https://www.investopedia.com/terms/o/option.asp)

## Terminology

- Financial instruments that give the right, but not the obligation, to buy or sell an underlying asset at a set strike price; offers investors the ability to leverage positions/hedge against risks
- American options - can be exercised any time before expiration; European options - can only be exercised at expiration 
- Call options: allow holder to buy asset at a stated price within specific time frame; has bullish buyer + bearish seller
- Put options: allow holder to sell asset at a stated price within a specific time frame; has bearish buyer + bullish seller
- At-the-money (ATM): strike price is exactly equal to an underlying asset price (delta = 0.5)
- In-the-money (ITM): for call, strike price is below current underlying price; for put, strike price is above current underlying price (intrinsic value, delta > 0.5)
- Out-of-the-money (OTM): for call, strike price is above current underlying price, for put, strike price is below current underlying price (extrinsic value, delta <0.5)
- Premium: price paid for option in the market
- Strike price: price which one can buy/sell the underlying (a.k.a. exercise price)
- Underlying: the security which the option is based on
- Exercise: when contract owner exercises their right to buy/sell at strike price
- Expiration: date which the contract expires a.k.a. the expiry


## Calls & Puts

### Buying Calls

- Allow holder to buy underlying at stated strike price by expiry, but not obligated to
- Risk to buyer is limited to premium paid
- Suppose buyers are bullish on a stock and predict share price will rise above strike price before expiration. If this happens, investor can buy the stock at strike price, then immediately sell stock at current market value for profit.
- Profit for one options contract = (market price - strike price - premium) * 100, assuming each contract has 100 shares
- If price doesn't move above strike price before expiration, holder isn't required to buy shares, but the option is worthless and premium paid is lost


### Selling Calls
- Buyer pays premium to writer/seller, which is the max profit when selling a call option. Seller expects stock price to fall or stay near strike price.
- If market share price <= strike price before expiry, option expires worthless for buyer --> seller makes profit by pocketing premium
- If market share price > strike price at expiry, seller must sell shares to buyer at that lower strike price. Seller must either sell ahres from own portfolio holdings or buy stock at market price to sell to buyer. This loss depends on cost basis of shares required to cover, subtracting away premium earned.
- Call buyer only at risk of losing premium, seller faces infinite risk (if stock price continuously rises)


### Buying Puts
- Allow holder to sell underlying at stated strike price by expiry, but not obligated to
- Buyer believes stock market price will fall below strike price before expiry; if this happens, the investor will exercise the put and sell shares at option's high strike price. If they want to replace their holdings, they can buy them back on open market.
- Profit for one options contract = (strike price - market price - premium) * 100, assuming each contract has 100 shares
- Value of holding of a put option increases as undelrying stock price decreases
- Buyer loss is limited to premium paid if expires worthlessly


### Selling Puts
- Buyer pays premium to writter/seller, which is the max profit when selling a put option. Seller expects stock price to increase or stay the same. 
- If underlying stock price closes below strike price, writer is obligated to buy shares of underlying stock at the same price. Seller is forced to purchase shares at strike price at expiry (above current market price).
- Loss depends on how much shares depreciate; can hold on to shares and hope stock price rises above purchase price or sell and take the loss. 


## Greeks

### Delta ($\Delta$)
- How much an option's price changes for $1 change in underlying asset's price; price sensitivity of the option relative to underlying
- Call has range 0 to 1, put has range 0 to -1
- Indicates hedge ratio needed for delta-neutral position e.g. purchase standard American call option w/ delta = x, need to sell 100x shares of stock to be fully hedged


### Theta ($\Theta$)
- Rate of change between option price and time (time sensitvity, time decay); indicates amount an option's price would decrease as time to expiration decreases e.g. long option w/ $\Theta = x$, option price decreases by $x$ dollars per day.
- Higher for ATM options, lower for ITM or OTM; options closer to expiration have accelerating time decay
- Long calls/puts usually have $\Theta <> 0$, short calls/puts $\Theta > 0$


### Gamma ($\Gamma$)
- Rate of change of between $\Delta$ and price of underlying asset (second-order price sensitivity); indicates amount $\Delta$ would change for $1 move in underlying
- Used to determine stability of option's $\Delta$; high $\Gamma$ means $\Delta$ changes rapidy, so option's price can accelerate in movement
- Higher for ATM otpions, lower for ITM or OTM, increasing as expiration nears


### Vega ($V$)
- Rate of change between option's value and underlying's implied volatility (sensitivity to volatility); indicates amount an option's price changes for 1% change in IV
- At its max for ATM otpions that have longer times until expiration, meaning option's value depnds strongly on volatility


### Rho ($\rho$)
- Rate of change between option's value and interest rate (sensitivity to interest rate); indicates amount an option's price changes for 1% change in interest rates
- Greatest for ATM options with long times until expiration

| Greek    | Measures                                          | Calls (Sign) | Puts (Sign)  | When It’s Largest / Most Important                   |
|----------|---------------------------------------------------|--------------|--------------|-------------------------------------------------------|
| Delta ($\Delta$) | Change in option premium per $1 change in underlying | + (0 → +1)   | – (0 → –1)   | ATM ~0.5; Deep ITM/OTM near ±1 or 0                    |
| Gamma ($\Gamma$) | Change in delta per $1 change in underlying        | +            | +            | Highest for ATM options near expiration               |
| Vega ($V$)  | Change in premium per 1% change in implied volatility | +            | +            | Highest for ATM, long-dated options                   |
| Theta ($\Theta$) | Time decay: premium change per day                  | –            | –            | Magnitude highest for ATM near expiration             |
| Rho ($\rho$)   | Change in premium per 1% change in interest rates   | +            | –            | Larger for longer-dated options                       |
