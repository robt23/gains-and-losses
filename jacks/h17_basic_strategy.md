# Blackjack H17 Basic Strategy & Deviations

## Basic Strategy

For now, we assume the most common house rules: dealer hits on soft 17 (A6), doubling after split is allowed, re-split is allowed, surrendering is allowed, and dealer pays 3:2. 
<br>
[Derived from Blackjack Apprenticeship](https://www.blackjackapprenticeship.com/wp-content/uploads/2024/09/H17-Basic-Strategy.pdf)

**Legend:**  
- H = Hit  
- S = Stand  
- D = Double (if allowed, else Hit)  
- Ds = Double (if allowed, else Stand)
- P = Split (if don't split, follow hit/stand rules)  
- U = Surrender

<br>

**Hard Totals:**

| Player / Dealer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|-----------------|---|---|---|---|---|---|---|---|----|---|
| 17–20           | S | S | S | S | S | S | S | S | S  | S |
| 16              | S | S | S | S | S | H | H | H | H  | H |
| 15              | S | S | S | S | S | H | H | H | H  | H |
| 14              | S | S | S | S | S | H | H | H | H  | H |
| 13              | S | S | S | S | S | H | H | H | H  | H |
| 12              | H | H | S | S | S | H | H | H | H  | H |
| 11              | D | D | D | D | D | D | D | D | D  | D |
| 10              | D | D | D | D | D | D | D | D | H  | H |
| 9               | H | D | D | D | D | H | H | H | H  | H |
| 8 or less       | H | H | H | H | H | H | H | H | H  | H |

<br>

**Soft Totals:**

| Player / Dealer | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|-----------------|---|---|---|---|---|---|---|---|----|---|
| A,9 (20)        | S | S | S | S | S | S | S | S | S  | S |
| A,8 (19)        | S | S | S | S | Ds| S | S | S | S  | S |
| A,7 (18)        | Ds| Ds| Ds| Ds| Ds| S | S | H | H  | H |
| A,6 (17)        | H | D | D | D | D | H | H | H | H  | H |
| A,5 (16)        | H | H | D | D | D | H | H | H | H  | H |
| A,4 (15)        | H | H | D | D | D | H | H | H | H  | H |
| A,3 (14)        | H | H | H | D | D | H | H | H | H  | H |
| A,2 (13)        | H | H | H | D | D | H | H | H | H  | H |


<br>

**Pairs:**

| Player / Dealer   | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|-------------------|---|---|---|---|---|---|---|---|----|---|
| A,A               | P | P | P | P | P | P | P | P | P  | P |
| 10,10             | S | S | S | S | S | S | S | S | S  | S |
| 9,9               | P | P | P | P | P | S | P | P | S  | S |
| 8,8               | P | P | P | P | P | P | P | P | P  | P |
| 7,7               | P | P | P | P | P | P | H | H | H  | H |
| 6,6               | P | P | P | P | P | H | H | H | H  | H |
| 5,5               | D | D | D | D | D | D | D | D | H  | H |
| 4,4               | H | H | H | P | P | H | H | H | H  | H |
| 3,3               | P | P | P | P | P | P | H | H | H  | H |
| 2,2               | P | P | P | P | P | P | H | H | H  | H |

<br>

**Surrenders:**
| Player / Dealer   | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|-------------------|---|---|---|---|---|---|---|---|----|---|
| 17                |   |   |   |   |   |   |   |   |    | U |
| 16                |   |   |   |   |   |   |   | U | U  | U |
| 15                |   |   |   |   |   |   |   |   | U  | U |
| 8,8               |   |   |   |   |   |   |   |   |    | U |

<br>

**Insurance/Even Money:**
DO NOT TAKE

---

## Deviations
Same assumptions as above from basic strategy.
<br>
[Derived from Blackjack Apprenticeship](https://www.blackjackapprenticeship.com/wp-content/uploads/2019/07/BJA_H17.pdf)

**Legend:**
- Numbers indicate the index that the true count must meet to deviation from basic strategy (same assumptions/rules apply)
- `+` after the index number indicates the deviation happens at that true count and above
- `-` after the index number indicates the devation happens at the true count and below
- 0- indicates the deviation happens at any negative running count
- 0+ indicates the deviation occurs at any positive running count

<br>

**Hard Totals:**

| Player / Dealer  | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------------------|---|---|---|---|---|---|---|---|----|---|
| 17               | S | S | S | S | S | S | S | S | S  | S |
| 16               | S | S | S | S | S | H | H | 4+| 0+ | 3+|
| 15               | S | S | S | S | S | H | H | H | 4+ | 5+|
| 14               | S | S | S | S | S | H | H | H | H  | H |
| 13               | -1-| S | S | S | S | H | H | H | H  | H |
| 12               | 3+| 2+| 0-| S | S | H | H | H | H  | H |
| 11               | D | D | D | D | D | D | D | D | D  | D |
| 10               | D | D | D | D | D | D | D | D | 4+ | 3+|
| 9                | 1+| D | D | D | D | 3+| H | H | H  | H |
| 8                | H | H | H | H | 2+| H | H | H | H  | H |

<br>

**Soft Totals:**

| Player / Dealer  | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------------------|---|---|---|---|---|---|---|---|----|---|
| A,9              | S | S | S | S | S | S | S | S | S  | S |
| A,8              | S | S | 3+| 1+| 0-| S | S | S | S  | S |
| A,7              | Ds| Ds| Ds| Ds| Ds| S | S | H | H  | H |
| A,6              | 1+| D | D | D | D | H | H | H | H  | H |
| A,5              | H | H | D | D | D | H | H | H | H  | H |
| A,4              | H | H | D | D | D | H | H | H | H  | H |
| A,3              | H | H | H | D | D | H | H | H | H  | H |
| A,2              | H | H | H | D | D | H | H | H | H  | H |

<br>

**Pairs:**

| Player / Dealer   | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|-------------------|---|---|---|---|---|---|---|---|----|---|
| A,A               | P | P | P | P | P | P | P | P | P  | P |
| 10,10             | S | S | S | S | S | S | S | S | S  | S |
| 9,9               | P | P | P | P | P | S | P | P | S  | S |
| 8,8               | P | P | P | P | P | P | P | P | P  | P |
| 7,7               | P | P | P | P | P | P | H | H | H  | H |
| 6,6               | P | P | P | P | P | H | H | H | H  | H |
| 5,5               | D | D | D | D | D | D | D | D | H  | H |
| 4,4               | H | H | P | P | P | H | H | H | H  | H |
| 3,3               | P | P | P | P | P | P | H | H | H  | H |
| 2,2               | P | P | P | P | P | P | H | H | H  | H |

<br>

**Surrenders:**

| Player / Dealer  | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|------------------|---|---|---|---|---|---|---|---|----|---|
| 17               |   |   |   |   |   |   |   |   |    | U |
| 16               |   |   |   |   |   |   | 4+|-1-| U  | U |  
| 15               |   |   |   |   |   |   |   | 2+| 0- |-1+|
| 14               |   |   |   |   |   |   |   |   |    |   |


<br>

**Insurance/Even Money:** Take at 3+
