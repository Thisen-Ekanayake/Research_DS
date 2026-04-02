# Member 5 — De Silva B.K.P.
## Task: Literature Review, Policy Framing & Lagged Dynamics

---

## Your role in the paper

You are the intellectual anchor of the study. While other members are running models, you are asking the harder questions: *What does this mean? Where does it fit in the literature? What should policymakers do with these findings?* The current paper's related work section is competent but thin — it leans heavily on four or five sources when the field is much richer. The policy section does not exist at all. The TLCC analysis in Figure 4 is presented without sufficient interpretation. All three of these are your responsibility, and fixing them will substantially raise the paper's contribution level.

---

## What you must deliver

| Output | Corresponds to |
|--------|---------------|
| Extended Section 2 (Related Work) — at minimum doubled in depth | Section 2 |
| Formal literature gap statement ("this is the first study to...") | Section 1 / Section 2 |
| TLCC analysis deepened — interpret the 2–3 quarter lag mechanism | Section 4.3 |
| Hysteresis analysis — does underemployment persist after shocks resolve? | Section 4.3 extended |
| Policy recommendations section (new Section 5 or expanded conclusion) | Section 5 / Conclusion |
| Comparison of Sri Lanka findings to international evidence | Section 2 / Discussion |
| Limitations section — fully expanded | Section 5 |

---

## Step-by-step guide

### Step 1 — Extend the literature review (Section 2)

The current Section 2 covers 5 sources in about half a page. A strong ACM short paper needs at minimum 10–12 well-integrated references. Here is the expanded structure you should build:

**Pillar A — Underemployment theory and measurement (3–4 sources)**

Start from first principles. Why does underemployment matter separately from unemployment? Synthesise:

- **Bell & Blanchflower (2018)** — already in the paper. Expand your treatment: their Underemployment Index shows hours-based slack persists long after headline unemployment recovers. Apply this specifically to Sri Lanka: unemployment returned to 3.8% by 2023 while underemployment remained elevated. This is precisely the Bell-Blanchflower dynamic.
- **ILO (2013) 19th ICLS Resolution** — already cited. Explain the formal ILO distinction between time-related underemployment (hours) and qualification-based mismatch (skills), and why both are needed for a complete picture.
- **Haider, S. (2001)** — "Earnings instability and earnings inequality of males in the United States: 1967–1991" (*Journal of Labor Economics*) — shows that underemployment predicts future earnings suppression even after re-employment.
- **Autor, D.H. (2014)** — "Skills, education, and the rise of earnings inequality among the 'other 99 percent'" (*Science*) — connects qualification mismatch to structural labour market polarisation. Relevant to Sri Lanka's graduate unemployment problem.

**Pillar B — Underemployment in developing and crisis economies (3–4 sources)**

- **OECD (2017)** — already cited. Expand: which specific structural drivers (services sector composition, youth participation) are the most persistent across the 29 countries? How does Sri Lanka compare on these dimensions?
- **Flek, V. & Mysikova, M. (2015)** — "Unskilled workers and the transition from unemployment to employment in Central Europe." *IZA Discussion Paper 8907* — shows that in post-crisis economies, low-skilled workers cycle through underemployment rather than recovering to full employment. Directly relevant to Sri Lanka's post-2022 trajectory.
- **Chowdhury, A. (2009)** — "Underemployment in developing Asia: Nature, causes and policy responses." *Asia-Pacific Development Journal*, 16(2), 1–32 — the most relevant regional reference. Covers Bangladesh, India, Indonesia, Thailand — provides the comparison baseline Sri Lanka lacks.
- **Islam, I. & Verick, S. (2011)** — "From the Great Recession to labour market recovery: Issues, evidence and policy options." ILO/Palgrave — covers the ILO employment response to sovereign debt crises. Directly analogous to Sri Lanka 2022.

**Pillar C — Sri Lanka labour market literature (3–4 sources)**

- **Pabasara & Silva (2025)** — already cited. Explicitly state the three limitations you address: (1) aggregate unemployment vs underemployment, (2) sample ends 2020 Q4, (3) linear VECM vs non-linear XGBoost. This is where you define your contribution gap.
- **Dissanayake (2026)** — already cited. Explain what they found and why their call for crisis-updated forecasts motivated this study.
- **Perera (2022) / Liyanapathirana & Samaraweera (2022)** — already cited (as [9]). Expand: their finding that agricultural sector employment is the primary source of seasonal inactive hours directly motivated the inclusion of the Agricultural Output Index.
- **Central Bank of Sri Lanka (2023)** — *Annual Report 2022*. The CBSL's own analysis of the 2022 economic crisis and its labour market effects. Provides official context for your structural break findings.

**Write the gap statement explicitly:**

At the end of Section 2, add a paragraph that begins: *"To the best of our knowledge, this study is the first to..."* and lists 4–5 specific firsts:
1. Apply SHAP-based ML interpretability to Sri Lankan underemployment.
2. Cover the full 2015–2025 crisis arc (Pabasara & Silva end at 2020).
3. Include remittances, agricultural output, part-time employment, and discouraged workers as predictors simultaneously.
4. Conduct a gender-disaggregated SHAP comparison (once Member 4 completes this).
5. Test structural breaks across 11 series simultaneously using both ZA and Bai-Perron PELT.

### Step 2 — Deepen the TLCC analysis (Section 4.3)

Figure 4 shows GDP growth and inflation peak at 2–3 quarter lags. This is currently presented as a correlation statistic. You need to interpret the *mechanism* behind the lag.

**The transmission mechanism you are describing:**

```
GDP contraction (Q0)
        ↓  [2–4 weeks]
Firms reduce orders, cut production
        ↓  [1–2 months]
Employers cut hours before cutting headcount (labour hoarding)
        ↓  [1 quarter]
Time-related underemployment rises (workers still employed but hours reduced)
        ↓  [2–3 quarters total]
LFS survey captures the elevated underemployment rate
```

Write this out as a narrative in Section 4.3. The 2–3 quarter lag is consistent with: (a) labour hoarding theory — employers retain workers at reduced hours before firing them outright, and (b) survey timing — LFS quarterly surveys lag economic events by up to one quarter.

**Hysteresis test:**

Does underemployment persist after the shock resolves? If GDP recovered in 2023 but underemployment remained elevated, that is evidence of labour market hysteresis.

```python
from statsmodels.tsa.stattools import adfuller

# Test if underemployment is stationary around a post-crisis mean
# If I(1) even after 2022, this suggests hysteresis (no mean reversion)
post_crisis = df.loc['2022Q3':, 'underemployment']
adf_result = adfuller(post_crisis, maxlag=2, regression='c')
print(f"ADF stat: {adf_result[0]:.3f}, p: {adf_result[1]:.4f}")
# If p > 0.05, underemployment is non-stationary in the post-crisis period → hysteresis
```

Compute the **impulse response function** from the VAR/ARDL model (Member 1 can provide this):
- A 1% GDP contraction → what is the predicted underemployment response over 8 quarters?
- Does underemployment return to pre-shock levels within 4 quarters, or does it remain elevated?

### Step 3 — Connect to Okun's Law

The current paper mentions Pabasara & Silva's VECM "confirms Okun's Law." But Okun's Law relates GDP growth to *unemployment* changes — not underemployment. You should:

1. Estimate Okun's coefficient for Sri Lanka on the underemployment series (not just unemployment).
2. Show that the underemployment Okun coefficient is larger in magnitude than the unemployment Okun coefficient — this is your evidence that underemployment is a more sensitive barometer.
3. Test whether the Okun coefficient changed post-2022 (this complements Member 1's interaction model).

```python
# Standard Okun's Law: ΔU = α + β × ΔY
# where ΔU = change in underemployment rate, ΔY = GDP growth

import statsmodels.formula.api as smf

# Okun's Law for underemployment
okun_u = smf.ols('d_underemployment ~ gdp_growth', data=df).fit(cov_type='HC3')
print(f"Underemployment Okun coeff: {okun_u.params['gdp_growth']:.3f}")

# Okun's Law for headline unemployment (comparison)
okun_ue = smf.ols('d_unemployment ~ gdp_growth', data=df).fit(cov_type='HC3')
print(f"Unemployment Okun coeff: {okun_ue.params['gdp_growth']:.3f}")
```

If the underemployment coefficient is, say, -0.8 vs the unemployment coefficient of -0.1, this quantitatively confirms the abstract's claim that underemployment is "a far more sensitive labour market barometer."

### Step 4 — Write the policy recommendations section

This is entirely new. Structure it around the top SHAP drivers:

**Recommendation 1 — Address structural informality (responds to SHAP finding: informal employment is the second-largest driver)**

- Sri Lanka's informal employment share has been rising since the 2017 structural break (ZA test). This predates the sovereign default, suggesting formalisation pressure is a structural, not cyclical, challenge.
- Specific policy: extend EPF/ETF (Employees' Provident Fund / Employees' Trust Fund) coverage to informal workers. Currently only formal sector employees are covered.
- Evidence base: ILO (2024) World Employment and Social Outlook shows countries that extended social protection to informal workers saw smaller underemployment increases during the 2008 and 2020 crises.

**Recommendation 2 — Targeted youth labour market interventions (responds to SHAP finding: Youth LFPR is the third-ranked driver)**

- Youth LFPR (15–24) is suggestive at nominal α=0.10 in Granger causality. This is consistent with qualification mismatch: young graduates cannot find employment matching their education, so they either exit the labour force or accept under-matched jobs.
- Specific policy: expand the "Graduate Entrepreneurship Programme" (already exists but underfunded) and create a formal job-matching platform between universities and the private sector.
- Reference: World Bank (2023) Sri Lanka Human Capital Review — recommends TVET (Technical and Vocational Education and Training) reform to reduce graduate mismatch.

**Recommendation 3 — Remittance stabilisation mechanisms (responds to SHAP finding: remittances partially cushion underemployment)**

- The 2021 remittance collapse (from $7.0 bn to $5.5 bn) preceded the peak underemployment quarter. Diaspora remittances provide a private-sector income buffer that compensates for lost working hours.
- Specific policy: CBSL should develop a formal remittance stabilisation mechanism — when remittance inflows fall below a threshold, activate a matching subsidy programme. This mirrors the IMF's Resilience and Sustainability Facility design principles.

**Recommendation 4 — Expand LFS measurement infrastructure (responds to the paper's core critique of headline unemployment as inadequate)**

- The 2022 LFS artefact (underemployment recorded at 2.3% due to survey concealment) shows that the current LFS design is vulnerable to social desirability bias during crises.
- Specific policy: DCS should adopt the ILO's composite Labour Underutilisation Framework (LU1–LU4 indicators) in all LFS quarterly reports. This would make underemployment a headline statistic alongside unemployment.

### Step 5 — Expand the limitations section

The current limitations paragraph is one short paragraph. Expand it to address:

1. **Small annual sample (n=10):** Explicitly state the degrees-of-freedom constraint and what it prevents (Johansen cointegration, full ARDL on annual data). State that the quarterly dataset (n≈43) substantially mitigates this.
2. **SHAP causality caveat:** SHAP values capture predictive association, not causal direction. A higher informal employment SHAP does not prove that informality *causes* underemployment — it may reflect a common third cause (e.g., economic shocks cause both informality and underemployment simultaneously).
3. **Missing variables:** Real wages and sectoral employment shares (manufacturing vs services vs agriculture) were not included due to quarterly data availability. Both are theoretically important channels.
4. **2022 LFS artefact:** Even with the quarterly workaround, the 2022 data quality is compromised. Results for that year should be interpreted with caution.
5. **External validity:** Sri Lanka is a small open economy with a specific crisis trajectory. Findings may not generalise to other developing countries without a similar sovereign default context.

---

## How to write this up

- Section 2 should now be 1–1.5 pages, structured around the three pillars above. Use subheadings if the venue allows them.
- The gap statement at the end of Section 2 should be a single, punchy paragraph — not a bulleted list. Reviewers read this to understand the contribution.
- Section 4.3 should include the transmission mechanism narrative as a textual flow diagram or step-by-step paragraph — not just the TLCC correlation numbers.
- The policy section should be concrete. Vague recommendations like "improve labour market policy" will not survive peer review. Name specific programmes, specific thresholds, specific agencies.
- The limitations section should demonstrate intellectual honesty — reviewers reward papers that acknowledge limitations before being asked to.

---

## Tips for stronger research

- **Read the CBSL 2022 Annual Report in full.** This is the official account of the crisis period. Your literature review should be in dialogue with it — where do your findings confirm the CBSL narrative, and where do they reveal something the CBSL missed?
- **The "barometer" framing is your central argument.** Every section should connect back to it: headline unemployment said 3.8%, but underemployment said 32.6%. Make that contrast vivid in the introduction, confirm it with data in Section 4.1, explain the mechanism in Section 4.3, and translate it into policy in Section 5.
- **Look for post-crisis comparators.** Greece (2010–2015 sovereign debt crisis), Argentina (2001 default), and Iceland (2008 banking crisis) all had detailed underemployment studies during their recovery periods. A 1–2 sentence comparison in Section 2 or the discussion will contextualise Sri Lanka's experience internationally.
- **Be precise about causality language throughout the paper.** Replace phrases like "X drives Y" with "X is associated with Y" or "X predicts Y" everywhere except where you have a genuine causal test (the Granger causality results). Reviewers at labour economics journals are strict about this.
- **The ACM short paper format (4 pages) is constraining.** Prioritise: (1) the gap statement, (2) the TLCC mechanism explanation, (3) one or two concrete policy recommendations. Everything else goes in an appendix or a longer journal version.

---

## Key references to read

- Bell, D.N.F. & Blanchflower, D.G. (2018). Underemployment in the US and Europe. *NBER Working Paper 24927*. — Read the full paper, especially Sections 4 (gender) and 5 (wage suppression).
- Chowdhury, A. (2009). Underemployment in developing Asia. *Asia-Pacific Development Journal*, 16(2). — The only regional survey paper; essential for Section 2 Pillar B.
- CBSL (2022). *Annual Report 2022*. Central Bank of Sri Lanka. — cbsl.gov.lk — Read Chapters 2 and 5.
- ILO (2024). *World Employment and Social Outlook: Trends 2024*. — Chapters 1 and 3 on developing country labour markets post-COVID.
- World Bank (2023). *Sri Lanka — Poverty and Equity Assessment*. World Bank Group. — Contains district-level and gender-disaggregated labour market data.
- Islam, I. & Verick, S. (eds.) (2011). *From the Great Recession to Labour Market Recovery*. Palgrave Macmillan / ILO. — Chapter 4 on crisis transmission mechanisms.

---

## Timeline suggestion

| Week | Action |
|------|--------|
| 1 | Read all 6 key references; annotate each for Section 2 contribution; draft gap statement |
| 2 | Write extended Section 2 (all three pillars); draft TLCC mechanism narrative for Section 4.3 |
| 3 | Estimate Okun coefficients (underemployment vs unemployment); draft policy recommendations |
| 4 | Expand limitations section; peer-review full paper draft with team; finalise Section 2 and Section 5 |
