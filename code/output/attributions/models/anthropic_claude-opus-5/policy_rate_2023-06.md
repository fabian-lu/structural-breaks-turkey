## Structural Break Analysis: Turkey Overnight Rate, June 2023

### What Changed

The series shifted from a low, drifting-down regime to a steeply rising, high-rate regime:

| | Before | After |
|---|---|---|
| Mean level | ~13.1% | ~40.1% |
| Trend | Decreasing | Sharply increasing |
| Volatility (SD) | 3.15 | 8.84 |

The detected step (+12.3 pp) captures the initial jump, but the post-break window mean is ~27 pp above the pre-break mean because the break initiated a *sustained tightening path* rather than a one-off level change. Volatility nearly tripled, reflecting large discrete policy steps rather than a smooth adjustment. This is a level-and-trend break plus a variance break — a textbook regime change.

### Likely Causes

The timing aligns almost exactly with Turkey's post-election policy pivot from unorthodox "low-rate" management back to conventional inflation targeting:

- **28 May 2023** – Presidential run-off won by Erdoğan, removing the electoral constraint on rate hikes.
- **3 June 2023** – Mehmet Şimşek appointed Treasury & Finance Minister, signalling a return to "rational" macro policy.
- **9 June 2023** – Hafize Gaye Erkan appointed CBRT Governor, replacing Şahap Kavcıoğlu.
- **Early June 2023** – Implicit FX defence eased; the lira depreciated sharply (roughly 19–20 to ~26 per USD within weeks), raising the urgency of monetary tightening.
- **22 June 2023** – First hike of the cycle: policy rate 8.5% → 15%, followed by consecutive increases (17.5% July, 25% August, 30% September, 35% October, 40% November, 42.5% December, 45% January 2024, 50% March 2024).
- Backdrop: CPI inflation re-accelerating from ~38% (mid-2023) toward ~65% (mid-2024), plus simplification/unwinding of macroprudential rules and lira-deposit (KKM) schemes that had previously suppressed and distorted interbank rates.

The pre-break level of ~13% above an 8.5% policy rate is consistent with interbank/call money trading well above the policy corridor midpoint under the prior regulatory regime; the break coincides with both the rate hikes and the normalisation of liquidity/collateral rules.

### Significance

- Statistically: 92.2% confidence, with simultaneous shifts in mean, trend, and variance — very unlikely to be noise or a seasonal artefact.
- Economically: one of the largest monetary policy regime shifts in any major emerging market in recent decades. It marks a durable change in the data-generating process, so any model estimated on pre-June-2023 data (Taylor rules, term-structure or pass-through models) should be re-estimated with a regime dummy or split sample.

**Caveat:** the attribution to specific appointments and announcements is based on well-documented public timing, but the exact decomposition between (a) policy rate hikes, (b) FX policy change, and (c) macroprudential normalisation cannot be separated from this single series alone.