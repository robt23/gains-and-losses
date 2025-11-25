# Poker Strategy

## Odds & Outs Calculations

- Ratio of amount in pot to amount of bet to call
- Whenever odds against < pot odds, do not call; whenever pot odds > odds against, call
- Example: $70 in pot (after opponent bets 20), $20 to call; pot odds 70:20 -> 3.5:1. 37 non-clubs in deck, 9 clubs; odds against 37-9 -> just over 4:1. NO CALL
- Betting the edge against opponent with flush draw (9 outs): you have top pair & top kicker with $100 in the pot - what is the minimum you should bet so that it is a mistake for him to call? 
<br>
Odds against for opponent are 37:9 --> ~4:1, (100 + x) / x = 4 for a toss up, where x = bet size
<br>
100 = 3x, x ~= 33. Anything greater than 33 will give them wrong odds to call
<br>
Use current pot size = (opponent odds against - 1) * bet size or bet size = current pot size / (opponent odds against - 1)
- Handy outs table:

| Number of Outs | Drawing Hand                                              |
|----------------|-----------------------------------------------------------|
| 4              | 2 pair, needing a full house; or inside straight draw     |
| 6              | 2 overcards needing to make a pair                        |
| 8              | Open-ended straight draw                                  |
| 9              | Flush draw                                                |
| 11             | Flush draw plus a pair needing to improve to trips        |
| 12             | Flush draw plus inside straight draw                      |
| 15             | Flush draw plus open-ended straight draw                  |

14 outs makes even money against a better hand; will always be getting better than even pot odss from any bet. 

| Bet Size   | Pot Odds |
|------------|----------|
| 2x pot     |   1.5:1  |
| full pot   |   2:1    |
| 3/4 pot    |   2.5:1  |
| 1/2 pot    |   3:1    |
| 1/4 pot    |   5:1    |

## Hands & Betting

- Premium hands: AA, KK, QQ, AK - play with raise from any spot on the table
- Solid hands: - JJ, TT, 99, 88, 77, AQs, AJs - play with raise if first to act, call if later spots on the table; almost never want to re-raise pre-flop
- Speculative hands: 66, 55, 44, 33, 22, suited aces, suited connectors - only raise in late position when action folds to you (in hijack, cutoff, button)
- Suited cards + small pairs go up in value when deep stacked; position is more important when deep stacked
- Premium hands go down in value when deep stacked