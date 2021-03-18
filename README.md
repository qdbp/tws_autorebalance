# tws_autorebalance

an automatic rebalancing bot for IB TWS

## Background

Like every healthy human being, I want to get rich. That's why I wrote this bot.

More?

Alright, so. Rebalancing is good, but doing it manually is a pain in the ass, so
this bot rebalances continuously.

Also, you don't get rich by just rebalancing -- you have to buy low and sell
high. Because I don't pretend I have any sort of edge or view that would let me
beat the market, I implement this in the most brain-dead way possible: tap
margin in proportion to drawdown from all-time-high. Shout-out to JPow + IBKR
for keeping those margin rates so juicily low, which is the only thing that
makes margin use for a retail schmuck like me remotely feasible.

What's the catch? As you can
see [here](https://github.com/qdbp/tws_autorebalance/blob/master/src/analysis/notebooks/util_investigation.ipynb)
, if drawdowns exceed around 60% I start liquidating at a huge loss (and am also
in huge debt). That's gg. Also, a rate hikes into a deep drawdown would turn the
negative cash balance into lava. There's probably other shit I haven't bothered
to think about. Other than that, it's free money.

## Roadmap

The architecture of this codebase is garbage because I didn't bother separating
the API logic from the rebalancing logic. For this reason this bot is impossible
to backtest -- or test, at all, when not connected to TWS during live market
hours.

While I personally find testing half-baked algorithm changes live in my main
account exciting, I can see why it's not everyone's jam. Flying blind vis-a-vis
even the most rudimentary backtest is also annoying.

For this reason, I'm currently rewriting this bot -- in Kotlin, because I'm sick
of Python and want to pad my portfolio.

Stay tuned.
