# 🛡️ Noon Fraud Data Lab

An interactive Streamlit dashboard for a synthetic food-delivery fraud dataset (UAE/GCC). It answers three questions:
**How close is the GAN-generated data to Noon's data? Can a fraud model trained on it be trusted? What does one transaction look like to that model?**

## Background

| Version | What changed | Gemini discriminator score |
|---|---|---|
| V1 | First WGAN-GP draft. Fraud prevalence 8.14%, far too high | 8.5 / 10 |
| V2 | Fraud rows dropped (not relabelled) to 2.3% prevalence | not provided |
| V3 | Fraud class retrained on Noon's synthetic-fraud pattern | not provided |
| V4 | Legitimate class retrained on Noon's reference traffic | 9.9 / 10 |

Scores for V1 and V4 are the ones reported by the project author. Edit or add V2/V3 in the sidebar (*Gemini discriminator scores*).

## Features

1. **Dynamic uploader (sidebar)**: upload *My Synthetic Dataset* and *Noon Reference Dataset*. With no upload, the app falls back to `FINAL_dataset_for_discriminator_4.csv` and `C_REFEREE_real_test.csv`. An optional third uploader replaces `B_NOON_synthetic_fraud.csv` for Tab 3.
2. **Tab 1, Dataset comparison & GAN story**: side-by-side metrics, version timeline, build-log charts, overlapping Plotly distributions (order amount, account age, distance), per-column distance bars, correlation-gap heatmap.
3. **Tab 2, What-if simulator**: type the order amount, set every other feature, and a Random Forest (or gradient boosting) trained in memory returns a live 0-100% risk, a LEGITIMATE / SUSPICIOUS / FRAUDULENT badge, a written explanation, per-feature drivers, an ops playbook (block / OTP-3DS / auto-approve), plus sensitivity curves and an hour × amount risk map.
4. **Tab 3, GAN vs GAN**: radar chart, pattern-gap bars, peak fraud hours (0-4 AM shaded), payment mix, Wilson 95% intervals, and a "fake-vs-real" test with a Noon-vs-Noon control.
5. **Tab 4, Column deep dive**: pick any column; mean/median/min/max/std side by side, distribution + box plot + ECDF, a discriminator note, and a fraud-vs-legit separation check.
6. **Tab 5, Professor's toolkit vs mine**: compares `A_eval_helper.py` and the professor's `app.py` to this project's classifier, runs both models head-to-head on the Noon reference set, and exports a `fraud_model.joblib` compatible with the professor's `app.py`.

## Honest metrics (computed live, not hard-coded)

* **Pattern-share similarity** = 100 − mean absolute gap (percentage points) across 14 fraud patterns. It is an average of shares, not proof the two GANs are indistinguishable.
* **Fake-vs-real accuracy** uses the professor's Random Forest idea, class-balanced, with a Noon-vs-Noon control (0.50 ≈ indistinguishable).
* Noon's reference file contains very few fraud rows (65 in the shipped file), so fraud-side comparisons are noisy. Intervals are shown for that reason.
* Models are trained on synthetic data, so scores are an upper bound on real-world performance.

## Repository layout

```
app.py
requirements.txt
README.md
.streamlit/config.toml
data/
  FINAL_dataset_for_discriminator_4.csv
  C_REFEREE_real_test.csv
  B_NOON_synthetic_fraud.csv
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (keep the CSVs in `data/`).
2. Go to <https://share.streamlit.io>, choose **New app**, pick the repo and set the main file to `app.py`.
3. Deploy. First load trains the models (about 15 s); results are cached afterwards.

## Expected CSV schema

`order_amount, account_age_days, hour_of_day, num_orders_last_7d, payment_method (card|wallet|cod|applepay), delivery_distance_km, discount_applied, prev_refunds, device_change, location_mismatch, is_fraud`. Extra columns such as `row_id` are ignored; rows with missing values are dropped.

## Notes

* Treat uploaded files as data only; the app never executes anything from them.
* The ops playbook in Tab 2 is illustrative, not Noon's real policy.
