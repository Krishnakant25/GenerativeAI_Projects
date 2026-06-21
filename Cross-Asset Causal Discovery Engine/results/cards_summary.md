# Layer-2 hypothesis cards — validation summary

**run_id:** `run_20260619_133653_683874bb`
**Model:** `llama3.1:8b-instruct-q4_0` (local Ollama, temperature 0.0, fixed seed)
**Cards generated:** 106 (one per significant candidate)
**Wall time:** 12.4 min

## Counts per plausibility flag
| Flag | Count |
|---|---|
| PLAUSIBLE_KNOWN_MECHANISM | 3 |
| PLAUSIBLE_NOVEL | 5 |
| LIKELY_SPURIOUS | 98 |
| MECHANISM_MISMATCH | 0 |
| PARSE_FAILED | 0 |

> The plausibility flag is a **heuristic filter, not validation**. An LLM can generate a confident, fluent mechanism for a statistically spurious relationship; this layer is designed to resist that, not to assume it away. Every card above still carries its corrected p-value.

## Mechanism-hallucination hardening (found → diagnosed → fixed)
Review of the first run found a real defect: the candidate `XLF`→`CL=F` (Financials ETF → Crude Oil) was flagged **plausible_novel** with the channel *'oil price -> input costs -> airline margins'* — a textbook channel for a DIFFERENT pair (oil → airlines), pattern-matched onto XLF/CL=F on the shared word 'oil'. It was **systemic**: re-validating the 10 previously-plausible cards, 8 of 10 channels changed and the same 'airline margins' phrase had been pasted onto `^NSEI`→`XLE` too (where airlines are irrelevant), while the one pair it genuinely fit (`XLE`→`JETS`) kept it.

**Fixed in two layers:** (1) the prompt now binds the candidate's two real asset names into the channel format and sanctions 'no clean mechanism for these two assets' (null + likely_spurious); (2) a structural backstop in the validator re-flags any channel that does not name both endpoints as `MECHANISM_MISMATCH` after one corrective retry. Final `MECHANISM_MISMATCH` count: 0 — every retained channel names its own pair. (Only the 10 previously-plausible cards were re-validated under the fix; the 96 already-spurious cards are unchanged.)

**Full flag-change breakdown (7 of the 10 changed flag):**

| Pair | Old → New | Read |
|---|---|---|
| `XLF`→`CL=F` | novel → **spurious** | ✅ killed hallucinated channel (oil→airline margins, wrong pair) |
| `^NSEI`→`XLE` | known → novel | ✅ old 'known' rested on the same airline hallucination; now honest |
| `^IXIC`→`^GSPC` | novel → **spurious** | ✅ Nasdaq/S&P share US equity beta → common-driver; declined to manufacture |
| `CL=F`→`XLF` | known → novel | ✅ generic old channel; oil→financials is loose → novel is more honest |
| `CL=F`→`^IXIC` | known → novel | ✅ 'input cost channel' was nonsensical for Nasdaq; novel is honest |
| `XLE`→`JPY=X` | known → novel | ◎ defensible mild conservatism (Japan oil-import → yen is real) |
| `XLE`→`^TNX` | known → novel | ⚠️ BORDERLINE over-correction: energy→inflation→long rates IS textbook; demoting known→novel is mild over-conservatism (still surfaced, not buried) |

**Honest caveat — not a clean win.** No card was wrongly demoted to *spurious* (both spurious flips are defensible), but `XLE`→`^TNX` is a genuine borderline: a textbook energy→inflation→rates channel was demoted known→novel, i.e. the hardened prompt is now slightly *over*-conservative on at least one defensible mechanism. It still surfaces as plausible_novel (not buried as spurious). Also: because the PROMPT itself changed, all 10 cards were fully re-generated, so these flag shifts conflate the targeted mismatch fix with general re-rating drift — they cannot be cleanly separated.

## Most confident PLAUSIBLE_KNOWN_MECHANISM cards
- **Energy Sector ETF → USD/INR** (`XLE`→`INR=X`) — channel: *Energy Sector ETF -> Oil Price Volatility -> USD/INR*
  - stat: corrected p = 9.87e-09, lag 1d, r = -0.146, in_graph = False
  - LLM confidence: 0.70
  - mechanism: The relationship between the Energy Sector ETF and USD/INR could be driven by changes in global oil prices, which are reflected in the energy sector's performance. As oil prices fluctuate, they can impact the value of the US dollar relative to other currencies like the Indian rupee.
  - caveats:
    - This relationship is episodic, suggesting it may be driven by specific events or market conditions rather than a persistent underlying mechanism.
    - The PC graph rejection suggests that other assets may mediate this relationship, which could indicate confounding or mediation.
    - PC rejected this as a *direct* edge; The strong predictive signal from the Energy Sector ETF to USD/INR is likely being mediated by a third asset, such as oil prices or global risk sentiment, which are not directly included in the PC graph. This would explain why the PC algorithm rejected this edge as a direct link.
    - Confounder check: It's possible that changes in global risk sentiment or US dollar strength could be driving this relationship, rather than a direct economic link between the energy sector and USD/INR.
- **Crude Oil → Energy Sector ETF** (`CL=F`→`XLE`) — channel: *Crude Oil -> Production Costs for Energy Companies -> Energy Sector ETF*
  - stat: corrected p = 1.26e-05, lag 3d, r = +0.062, in_graph = True
  - LLM confidence: 0.70
  - mechanism: The relationship between Crude Oil and Energy Sector ETF could be driven by changes in the cost of production for energy companies, which are reflected in the price of crude oil.
  - caveats:
    - The relationship is episodic, suggesting that it may be influenced by specific events or market conditions.
    - Further analysis would be needed to determine whether a common driver is indeed the cause of this relationship.
    - Confounder check: It's possible that changes in interest rates or the US dollar could be driving this relationship, as they can impact both crude oil prices and energy sector performance.
- **Energy Sector ETF → US Global Jets (Airlines) ETF** (`XLE`→`JETS`) — channel: *Energy Sector ETF -> Fuel Prices -> US Global Jets (Airlines) ETF*
  - stat: corrected p = 5.22e-04, lag 5d, r = +0.108, in_graph = False
  - LLM confidence: 0.70
  - mechanism: The relationship between Energy Sector ETF and US Global Jets (Airlines) ETF could be driven by changes in fuel prices, which affect the profitability of airlines and are reflected in energy sector stocks. As fuel prices rise or fall, both sectors tend to move together.
  - caveats:
    - This relationship is episodic and not permanent, which suggests it may be driven by specific events or market conditions rather than a fundamental link between the two sectors.
    - The PC graph rejection could be due to the presence of a third asset (e.g. fuel prices) that mediates the relationship between Energy Sector ETF and US Global Jets (Airlines) ETF.
    - PC rejected this as a *direct* edge; The conditional-independence test performed by PC likely rejected this edge because it found that once other assets were conditioned on, the relationship between Energy Sector ETF and US Global Jets (Airlines) ETF became independent. This suggests that a third asset (e.g. fuel prices) may be mediating the relationship between these two sectors.
    - Confounder check: It's possible that changes in global economic sentiment or interest rates could be driving this relationship, as both energy and airline stocks tend to move together during periods of economic growth or decline.

## LIKELY_SPURIOUS flags (98)
The model declined to endorse a mechanism for these — typically citing a common-driver confound (rates, the dollar, broad risk sentiment) or the relationship's episodic regime history. A sample of the most confidently-spurious:
- **S&P 500 → Nifty 50** (`^GSPC`→`^NSEI`)
  - stat: corrected p = 3.43e-42, lag 5d, r = +0.127, in_graph = False
  - LLM confidence: 0.80
  - mechanism: The S&P 500's predictive precedence over the Nifty 50 could be due to the influence of global risk sentiment, which tends to move both indices in tandem.
  - caveats:
    - The relationship is episodic, suggesting that it may be driven by external factors rather than a fundamental connection between the two indices.
    - The PC graph rejection suggests that the relationship may be mediated or confounded, which further supports the likelihood of spuriousness.
    - PC rejected this as a *direct* edge; The strong predictive signal from the S&P 500 to the Nifty 50 is likely rejected as a direct link by PC because it is mediated or confounded by a third asset, such as global risk sentiment.
    - Confounder check: A common driver such as broad risk sentiment could produce this statistical relationship without a direct economic link between the S&P 500 and Nifty 50.
- **USD/JPY → S&P 500** (`JPY=X`→`^GSPC`)
  - stat: corrected p = 7.52e-03, lag 1d, r = +0.066, in_graph = False
  - LLM confidence: 0.80
  - mechanism: The relationship between USD/JPY and S&P 500 could be driven by changes in global risk sentiment, which affects both the yen's value against the dollar and US stock market performance.
  - caveats:
    - USD/JPY is often influenced by yen carry trades, which can impact the US stock market through changes in investor sentiment
    - The relationship may be driven by other factors such as changes in global interest rates or economic conditions
    - PC rejected this as a *direct* edge; PC likely rejected a Granger-strong edge between USD/JPY and S&P 500 because the relationship is mediated through changes in global risk sentiment, which affects both assets.
    - Confounder check: A common driver such as changes in global risk sentiment could produce this statistical relationship without a direct economic link between USD/JPY and S&P 500.
- **10Y Treasury Yield → USD/JPY** (`^TNX`→`JPY=X`)
  - stat: corrected p = 7.09e-46, lag 1d, r = +0.331, in_graph = False
  - LLM confidence: 0.70
  - mechanism: The relationship between the 10Y Treasury Yield and USD/JPY could be driven by changes in global risk sentiment, which affects both US Treasury yields and the Japanese yen's value. When investors become more risk-averse, they tend to seek safer assets like US Treasuries, causing their prices to rise and yields to fall. This, in turn, can lead to a stronger USD/JPY exchange rate as investors sell JPY-denominated assets.
  - caveats:
    - The relationship is episodic, suggesting that it may be driven by specific events or market conditions rather than a persistent underlying mechanism.
    - The PC graph rejection suggests that there may be a mediating factor at play, which could further support the idea of spuriousness.
    - PC rejected this as a *direct* edge; The strong predictive signal between the 10Y Treasury Yield and USD/JPY is likely rejected as a direct link by PC because it may be mediated through changes in global risk sentiment or other sector factors, which are not directly captured by the Granger causality test.
    - Confounder check: A common driver, such as changes in global risk sentiment or a sector factor like the US dollar's value, could produce this statistical relationship without a direct economic link between the 10Y Treasury Yield and USD/JPY.
- **Gold → USD/JPY** (`GC=F`→`JPY=X`)
  - stat: corrected p = 7.28e-29, lag 1d, r = -0.263, in_graph = False
  - LLM confidence: 0.70
  - mechanism: The relationship between gold prices and USD/JPY exchange rates may be influenced by changes in global risk sentiment, which can impact both the value of gold and the yen's exchange rate. As investors become more risk-averse, they tend to buy safe-haven assets like gold, causing its price to rise. This increased demand for gold can lead to a stronger yen, as investors sell other currencies to purchase gold.
  - caveats:
    - The relationship is episodic, suggesting that it may be driven by specific events or market conditions rather than a fundamental economic mechanism.
    - The PC graph rejection suggests that the relationship may be mediated through other assets or factors.
    - PC rejected this as a *direct* edge; The strong predictive signal between gold prices and USD/JPY exchange rates is likely rejected as a direct link by the PC algorithm because it is mediated through changes in global risk sentiment, which affects both assets. This mediation pathway is not captured by the direct edge test used by Granger causality.
    - Confounder check: A common driver such as changes in global risk sentiment could produce this statistical relationship without a direct economic link between gold prices and USD/JPY exchange rates.
- **10Y Treasury Yield → EUR/USD** (`^TNX`→`EURUSD=X`)
  - stat: corrected p = 2.70e-07, lag 1d, r = -0.126, in_graph = False
  - LLM confidence: 0.70
  - mechanism: The relationship between the 10Y Treasury Yield and EUR/USD could be driven by changes in global risk sentiment, which affects both US interest rates and currency markets.
  - caveats:
    - The relationship is episodic, suggesting that it may be driven by specific events or market conditions rather than a persistent underlying mechanism.
    - The PC graph rejection suggests that the relationship may be mediated through other assets, which could indicate confounding or spuriousness.
    - PC rejected this as a *direct* edge; The strong predictive signal from the 10Y Treasury Yield to EUR/USD is likely rejected as a direct link by PC because it is mediated through changes in global risk sentiment, which affects both US interest rates and currency markets.
    - Confounder check: A common driver such as broad risk sentiment could produce this statistical relationship without a direct economic link between the 10Y Treasury Yield and EUR/USD.

## Canonical PC-rejection case: ^TNX → JPY=X
Granger-strong (corrected p = 7.09e-46, lag 1d, r = +0.331) yet **in_graph = False** — PC rejected it as a *direct* edge.

- **plausibility_flag:** likely_spurious
- **addresses_pc_rejection:** True
- **LLM confidence:** 0.70
- **mechanism:** The relationship between the 10Y Treasury Yield and USD/JPY could be driven by changes in global risk sentiment, which affects both US Treasury yields and the Japanese yen's value. When investors become more risk-averse, they tend to seek safer assets like US Treasuries, causing their prices to rise and yields to fall. This, in turn, can lead to a stronger USD/JPY exchange rate as investors sell JPY-denominated assets.
- **caveats:**
    - The relationship is episodic, suggesting that it may be driven by specific events or market conditions rather than a persistent underlying mechanism.
    - The PC graph rejection suggests that there may be a mediating factor at play, which could further support the idea of spuriousness.
    - PC rejected this as a *direct* edge; The strong predictive signal between the 10Y Treasury Yield and USD/JPY is likely rejected as a direct link by PC because it may be mediated through changes in global risk sentiment or other sector factors, which are not directly captured by the Granger causality test.
    - Confounder check: A common driver, such as changes in global risk sentiment or a sector factor like the US dollar's value, could produce this statistical relationship without a direct economic link between the 10Y Treasury Yield and USD/JPY.

## Spurious-control probe (honest)
A deliberately fabricated candidate — two economically-unrelated assets with an invented 'significant' statistic — fed through the SAME prompt, to check the model is not rubber-stamping everything as plausible.

- Fabricated candidate: `INR=X`→`NG=F` (invented corrected p = 1e-09)
- **Result flag:** likely_spurious (LLM confidence 0.50)
- **Mechanism the model offered:** The appreciation of the Indian Rupee against the US Dollar could lead to increased imports of natural gas, driving up prices.

✅ The model FLAGGED the nonsense as spurious — it did not rationalise it.
