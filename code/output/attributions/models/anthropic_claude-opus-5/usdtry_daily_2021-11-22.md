## Structural Break Analysis: USD/TRY, 22 November 2021

### What Changed

**Level shift:** The 30-day mean jumped from 9.63 to 13.06 — a ~36% depreciation of the lira in the space of a month. This is roughly 7.5 pre-break standard deviations, an extreme discontinuity even by emerging-market FX standards.

**Volatility regime shift:** Standard deviation nearly tripled (0.46 → 1.31). The rise in dispersion is arguably more informative than the level shift: it signals a transition from an orderly, trending depreciation to a disorderly, headline-driven regime.

**Trend change:** The pre-break window shows a persistent upward drift (gradual, policy-consistent lira weakening). Post-break the trend flattens while volatility explodes — characteristic of a currency that has overshot violently and is then subject to intervention, policy shocks and sharp mean-reverting swings rather than a smooth trend. Note that a "stable trend with high variance" over this specific window is partly a statistical artifact of two offsetting moves: a parabolic collapse into mid/late December followed by an abrupt retracement.

### Likely Drivers

The timing maps directly onto identifiable Turkish policy events (documented, not speculative):

- **18 November 2021:** The CBRT cut its policy rate by 100bp to 15%, the third consecutive cut, with CPI inflation running near 20% — deeply negative real rates.
- **22 November 2021 (the break date):** President Erdoğan publicly defended the easing cycle, framing it as an "economic war of independence" and reiterating his opposition to high interest rates. The lira fell roughly 15% intraday on 22–23 November, breaching 13 for the first time.
- **16 December 2021:** A further 100bp cut to 14% accelerated the collapse; USD/TRY spiked to ~18.4 on 20 December.
- **20 December 2021:** Announcement of the FX-protected lira deposit scheme (KKM), which triggered a violent ~25% single-day lira rally back toward 11–12.

The proximate cause is therefore a **credibility shock to monetary policy** — the market's repricing of the CBRT's reaction function once it became clear that rates would be cut *into* accelerating inflation. Contributing background factors: rising expectations of Fed tightening (Powell renominated 22 November 2021, hawkish pivot), depleted CBRT net FX reserves limiting intervention capacity, and a wide current-account/energy import bill.

### Significance

- The 84.9% confidence is likely understated by the post-break volatility explosion, which inflates the residual variance used in the test statistic. Economically, the break is unambiguous: no plausible continuation of the pre-break process generates a 36% level shift in one month.
- This is a **permanent regime change**, not a transient spike. The lira never returned to single digits; the pre-break parameters have no forecasting value thereafter.
- **Practical implication:** any model fitted on pre-November 2021 data (volatility, VaR, carry-trade, pass-through to CPI) must be re-estimated. The post-break window's parameters are also unreliable, as they span both the collapse and the KKM-driven rebound — a longer post-break sample is needed for stable estimates.