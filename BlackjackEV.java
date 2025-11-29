import os
import random
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class BlackjackEV:
    """
    Engine: Infinite-deck Monte-Carlo EV simulator for single-hand blackjack.
    Goal: Achieve convergence to basic strategy chart. 
    Rules:
        - Dealer hits soft 17 (H17)
        - Blackjack pays 3:2
        - Double allowed on any two cards
        - No surrender / no splits (yet)
    """

    def __init__(self, n_rollouts=10_000, max_depth=10, seed=42, n_workers=None):
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_workers = max(1, n_workers or (os.cpu_count() or 4))
        self._local = threading.local()

        # Card ranks (2-A) for an infinite deck; each equally likely
        self.cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

    # -----------------------------------------------------------
    # Utilities                         
    # -----------------------------------------------------------
    def _get_rng(self):
        rng = getattr(self._local, "rng", None)
        if rng is None:
            rng = random.Random(self.seed ^ threading.get_ident())
            self._local.rng = rng
        return rng

    def draw_card(self):
        return self._get_rng().choice(self.cards)

    def hand_value(self, cards):
        total = sum(cards)
        aces = cards.count(11)
        soft = False
        
        # calculate aces working in favor
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        # soft = True if there is an Ace counted as 11
        if 11 in cards and total <= 21:
            soft = True if total + 10 <= 21 else False
            
        return total, soft

    # -----------------------------------------------------------
    # Dealer play (H17)
    # -----------------------------------------------------------
    def dealer_play(self, upcard):
        dealer = [upcard, self.draw_card()]
        total, soft = self.hand_value(dealer)
        
        # Dealer blackjack check
        if len(dealer) == 2 and total == 21:
            return dealer
        
        while True:
            total, soft = self.hand_value(dealer)
            if total > 21:
                break
            if total < 17:
                dealer.append(self.draw_card())
                continue
            if total == 17 and soft:  # hit soft 17
                dealer.append(self.draw_card())
                continue
            break
        
        return dealer

    def _chunk_counts(self, total):
        base = total // self.n_workers
        remainder = total % self.n_workers
        return [base + (1 if i < remainder else 0) for i in range(self.n_workers)]

    def _repeat_worker(self, count, worker_fn):
        return [worker_fn() for _ in range(count)]

    def _run_parallel(self, iterations, worker_fn, parallel=True):
        if iterations <= 0:
            return []
        if not parallel or self.n_workers == 1 or iterations < self.n_workers * 4:
            return [worker_fn() for _ in range(iterations)]

        counts = self._chunk_counts(iterations)
        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._repeat_worker, count, worker_fn)
                for count in counts
                if count > 0
            ]
            for fut in futures:
                results.extend(fut.result())
        return results

    # -----------------------------------------------------------
    # Settle result
    # -----------------------------------------------------------
    def settle(self, player_cards, dealer_cards, doubled=False):
        player_total, _ = self.hand_value(player_cards)
        dealer_total, _ = self.hand_value(dealer_cards)
        bet = 2 if doubled else 1

        # Natural blackjack
        if len(player_cards) == 2 and player_total == 21:
            if len(dealer_cards) == 2 and dealer_total == 21:
                return 0
            return 1.5 * bet

        if player_total > 21:
            return -1 * bet
        if dealer_total > 21:
            return 1 * bet
        if player_total > dealer_total:
            return 1 * bet
        if player_total < dealer_total:
            return -1 * bet
        return 0

    # -----------------------------------------------------------
    # EV(stand)
    # -----------------------------------------------------------
    def EV_stand(self, player_total, player_soft, dealer_up, rollouts=None, parallel=True):
        """Expected value from standing immediately."""
        rollouts = rollouts if rollouts is not None else self.n_rollouts

        def simulate_once():
            dealer_hand = self.dealer_play(dealer_up)
            dealer_total, _ = self.hand_value(dealer_hand)
            if player_total > 21:
                return -1
            if dealer_total > 21:
                return 1
            if player_total > dealer_total:
                return 1
            if player_total < dealer_total:
                return -1
            return 0

        results = self._run_parallel(rollouts, simulate_once, parallel=parallel)
        return np.mean(results)

    # -----------------------------------------------------------
    # EV(hit) – recursive Monte Carlo
    # -----------------------------------------------------------
    def EV_hit(self, cards, dealer_up, depth=0):
        total, soft = self.hand_value(cards)
        if total > 21:
            return -1.0
        if depth >= self.max_depth:
            return self.EV_stand(total, soft, dealer_up)

        total_reward = 0.0
        samples = max(10, self.n_rollouts // 100)  # fewer per recursion
        for _ in range(samples):
            c = self.draw_card()
            new_cards = cards + [c]
            new_total, new_soft = self.hand_value(new_cards)
            if new_total > 21:
                total_reward += -1
                continue
            # recursively decide optimally between hitting again or standing
            ev_hit_next = self.EV_hit(new_cards, dealer_up, depth + 1)
            ev_stand_next = self.EV_stand(new_total, new_soft, dealer_up, parallel=False)
            total_reward += max(ev_hit_next, ev_stand_next)
        return total_reward / samples

    # -----------------------------------------------------------
    # EV(double)
    # -----------------------------------------------------------
    def EV_double(self, cards, dealer_up):
        samples = max(50, self.n_rollouts // 10)

        def simulate_once():
            c = self.draw_card()
            new_cards = cards + [c]
            dealer_hand = self.dealer_play(dealer_up)
            return self.settle(new_cards, dealer_hand, doubled=True)

        results = self._run_parallel(samples, simulate_once)
        return np.mean(results)

    # -----------------------------------------------------------
    # Evaluate all actions
    # -----------------------------------------------------------
    def evaluate(self, cards, dealer_up):
        total, soft = self.hand_value(cards)
        ev_stand = float(self.EV_stand(total, soft, dealer_up))
        ev_hit = float(self.EV_hit(cards, dealer_up))
        ev_double = float(self.EV_double(cards, dealer_up))

        best_action = max(
            [("Stand", ev_stand), ("Hit", ev_hit), ("Double", ev_double)],
            key=lambda x: x[1],
        )[0]

        return {
            "EV(Stand)": ev_stand,
            "EV(Hit)": ev_hit,
            "EV(Double)": ev_double,
            "Best Action": best_action,
        }


sim = BlackjackEV(n_rollouts=10000)
res = sim.evaluate([3, 8], dealer_up=9)
print(res)
