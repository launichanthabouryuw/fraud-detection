"""
Integration tests for features.py and analyze_fraud.py.

These tests load the real CSV files and validate the end-to-end pipeline
against known business metrics: fraud labels, chargeback rates, and loss dollars.
"""
from pathlib import Path

import pandas as pd
import pytest

from analyze_fraud import score_transactions, summarize_results
from features import build_model_frame

DATA_DIR = Path(__file__).resolve().parent

KNOWN_FRAUD_IDS = {50003, 50006, 50008, 50011, 50013, 50014, 50015, 50019}
KNOWN_CLEAN_IDS = {50001, 50002, 50004, 50005, 50007, 50009, 50010, 50012, 50016, 50017, 50018, 50020}


@pytest.fixture(scope="module")
def raw_data():
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    chargebacks = pd.read_csv(DATA_DIR / "chargebacks.csv")
    return accounts, transactions, chargebacks


@pytest.fixture(scope="module")
def scored(raw_data):
    accounts, transactions, _ = raw_data
    return score_transactions(transactions, accounts)


@pytest.fixture(scope="module")
def summary(scored, raw_data):
    _, _, chargebacks = raw_data
    return summarize_results(scored, chargebacks)


# ---------------------------------------------------------------------------
# build_model_frame
# ---------------------------------------------------------------------------

def test_build_model_frame_includes_transaction_columns():
    txns = pd.DataFrame([{"transaction_id": 1, "account_id": 99, "amount_usd": 50.0,
                           "device_risk_score": 20, "is_international": 0,
                           "velocity_24h": 1, "failed_logins_24h": 0}])
    accts = pd.DataFrame([{"account_id": 99, "prior_chargebacks": 1,
                            "account_age_days": 100, "kyc_level": "full", "is_vip": "N"}])
    result = build_model_frame(txns, accts)
    for col in ["transaction_id", "amount_usd", "device_risk_score", "is_international"]:
        assert col in result.columns


def test_build_model_frame_includes_account_columns():
    txns = pd.DataFrame([{"transaction_id": 1, "account_id": 99, "amount_usd": 50.0,
                           "device_risk_score": 20, "is_international": 0,
                           "velocity_24h": 1, "failed_logins_24h": 0}])
    accts = pd.DataFrame([{"account_id": 99, "prior_chargebacks": 1,
                            "account_age_days": 100, "kyc_level": "full", "is_vip": "N"}])
    result = build_model_frame(txns, accts)
    for col in ["prior_chargebacks", "account_age_days", "kyc_level", "is_vip"]:
        assert col in result.columns


def test_build_model_frame_prior_chargebacks_comes_from_accounts():
    txns = pd.DataFrame([{"transaction_id": 1, "account_id": 99, "amount_usd": 50.0,
                           "device_risk_score": 20, "is_international": 0,
                           "velocity_24h": 1, "failed_logins_24h": 0}])
    accts = pd.DataFrame([{"account_id": 99, "prior_chargebacks": 3,
                            "account_age_days": 100, "kyc_level": "full", "is_vip": "N"}])
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["prior_chargebacks"] == 3


def test_build_model_frame_unmatched_account_produces_nan():
    txns = pd.DataFrame([{"transaction_id": 1, "account_id": 999, "amount_usd": 50.0,
                           "device_risk_score": 20, "is_international": 0,
                           "velocity_24h": 1, "failed_logins_24h": 0}])
    accts = pd.DataFrame([{"account_id": 1, "prior_chargebacks": 0,
                            "account_age_days": 100, "kyc_level": "full", "is_vip": "N"}])
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["prior_chargebacks"] != result.iloc[0]["prior_chargebacks"]  # NaN check


def test_build_model_frame_row_count_matches_transactions(raw_data):
    accounts, transactions, _ = raw_data
    result = build_model_frame(transactions, accounts)
    assert len(result) == len(transactions)


def test_build_model_frame_all_accounts_matched(raw_data):
    accounts, transactions, _ = raw_data
    result = build_model_frame(transactions, accounts)
    assert result["prior_chargebacks"].isna().sum() == 0


# ---------------------------------------------------------------------------
# score_transactions — fraud label correctness
# ---------------------------------------------------------------------------

def test_no_known_fraud_labeled_low(scored):
    fraud_rows = scored[scored["transaction_id"].isin(KNOWN_FRAUD_IDS)]
    low_fraud = fraud_rows[fraud_rows["risk_label"] == "low"]
    assert len(low_fraud) == 0, (
        f"Fraud transactions incorrectly labeled low: {low_fraud['transaction_id'].tolist()}"
    )


def test_high_risk_fraud_labeled_high(scored):
    # These 7 transactions have enough compounding signals to reach the high threshold
    high_risk_fraud = {50003, 50006, 50011, 50013, 50014, 50015, 50019}
    fraud_rows = scored[scored["transaction_id"].isin(high_risk_fraud)]
    assert (fraud_rows["risk_label"] == "high").all(), (
        f"Expected high label for: {fraud_rows[fraud_rows['risk_label'] != 'high']['transaction_id'].tolist()}"
    )


def test_risk_score_column_is_present(scored):
    assert "risk_score" in scored.columns


def test_risk_label_column_is_present(scored):
    assert "risk_label" in scored.columns


def test_risk_scores_within_valid_range(scored):
    assert scored["risk_score"].between(0, 100).all()


def test_risk_labels_are_valid_values(scored):
    assert set(scored["risk_label"].unique()).issubset({"low", "medium", "high"})


def test_all_transactions_are_scored(scored, raw_data):
    _, transactions, _ = raw_data
    assert len(scored) == len(transactions)


# ---------------------------------------------------------------------------
# summarize_results — output structure
# ---------------------------------------------------------------------------

def test_summary_has_required_columns(summary):
    required = {"risk_label", "transactions", "total_amount_usd",
                "avg_amount_usd", "chargebacks", "chargeback_rate"}
    assert required.issubset(set(summary.columns))


def test_summary_chargeback_rate_no_nan(summary):
    assert summary["chargeback_rate"].isna().sum() == 0


def test_summary_chargebacks_no_nan(summary):
    assert summary["chargebacks"].isna().sum() == 0


def test_summary_chargebacks_is_integer(summary):
    assert summary["chargebacks"].dtype in (int, "int64", "int32")


def test_summary_total_transactions_matches_dataset(summary, raw_data):
    _, transactions, _ = raw_data
    assert summary["transactions"].sum() == len(transactions)


def test_summary_total_chargebacks_matches_known_fraud(summary):
    assert summary["chargebacks"].sum() == len(KNOWN_FRAUD_IDS)


# ---------------------------------------------------------------------------
# summarize_results — chargeback rate by band
# ---------------------------------------------------------------------------

def test_high_band_chargeback_rate_is_1(summary):
    high = summary[summary["risk_label"] == "high"]
    assert len(high) == 1, "Expected a 'high' risk band in summary"
    assert high.iloc[0]["chargeback_rate"] == 1.0


def test_low_band_chargeback_rate_is_0(summary):
    low = summary[summary["risk_label"] == "low"]
    assert len(low) == 1, "Expected a 'low' risk band in summary"
    assert low.iloc[0]["chargeback_rate"] == 0.0


def test_high_band_chargeback_count(summary):
    high = summary[summary["risk_label"] == "high"]
    assert high.iloc[0]["chargebacks"] == 7


def test_low_band_chargeback_count_is_zero(summary):
    low = summary[summary["risk_label"] == "low"]
    assert low.iloc[0]["chargebacks"] == 0


# ---------------------------------------------------------------------------
# summarize_results — fraud loss dollars
# ---------------------------------------------------------------------------

def test_high_band_total_amount_usd(summary):
    high = summary[summary["risk_label"] == "high"]
    assert pytest.approx(high.iloc[0]["total_amount_usd"], rel=1e-3) == 4234.98


def test_avg_amount_usd_equals_total_divided_by_count(summary):
    for _, row in summary.iterrows():
        expected_avg = row["total_amount_usd"] / row["transactions"]
        assert pytest.approx(row["avg_amount_usd"], rel=1e-6) == expected_avg
