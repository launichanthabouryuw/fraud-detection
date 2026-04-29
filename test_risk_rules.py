from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Zero-score baseline: all signals below every threshold."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_exact_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# ---------------------------------------------------------------------------
# score_transaction — device risk
# ---------------------------------------------------------------------------

def test_device_risk_below_40_adds_no_points():
    assert score_transaction(_base_tx(device_risk_score=39)) == 0


def test_device_risk_40_adds_10():
    assert score_transaction(_base_tx(device_risk_score=40)) == 10


def test_device_risk_mid_range_adds_10():
    assert score_transaction(_base_tx(device_risk_score=55)) == 10


def test_device_risk_70_adds_25():
    assert score_transaction(_base_tx(device_risk_score=70)) == 25


def test_device_risk_above_70_adds_25():
    assert score_transaction(_base_tx(device_risk_score=85)) == 25


def test_high_device_risk_adds_points():
    assert score_transaction(_base_tx(device_risk_score=75)) > score_transaction(_base_tx(device_risk_score=10))


# ---------------------------------------------------------------------------
# score_transaction — international
# ---------------------------------------------------------------------------

def test_international_adds_15():
    assert score_transaction(_base_tx(is_international=1)) == 15


def test_domestic_adds_no_international_points():
    assert score_transaction(_base_tx(is_international=0)) == 0


def test_international_adds_risk():
    assert score_transaction(_base_tx(is_international=1)) > score_transaction(_base_tx(is_international=0))


# ---------------------------------------------------------------------------
# score_transaction — amount
# ---------------------------------------------------------------------------

def test_amount_below_500_adds_no_points():
    assert score_transaction(_base_tx(amount_usd=499)) == 0


def test_amount_500_adds_10():
    assert score_transaction(_base_tx(amount_usd=500)) == 10


def test_amount_mid_range_adds_10():
    assert score_transaction(_base_tx(amount_usd=750)) == 10


def test_amount_1000_adds_25():
    assert score_transaction(_base_tx(amount_usd=1000)) == 25


def test_amount_above_1000_adds_25():
    assert score_transaction(_base_tx(amount_usd=2000)) == 25


def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_medium_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=600)) >= 10


# ---------------------------------------------------------------------------
# score_transaction — velocity
# ---------------------------------------------------------------------------

def test_velocity_below_3_adds_no_points():
    assert score_transaction(_base_tx(velocity_24h=2)) == 0


def test_velocity_3_adds_5():
    assert score_transaction(_base_tx(velocity_24h=3)) == 5


def test_velocity_mid_range_adds_5():
    assert score_transaction(_base_tx(velocity_24h=4)) == 5


def test_velocity_6_adds_20():
    assert score_transaction(_base_tx(velocity_24h=6)) == 20


def test_velocity_above_6_adds_20():
    assert score_transaction(_base_tx(velocity_24h=10)) == 20


def test_high_velocity_adds_risk():
    assert score_transaction(_base_tx(velocity_24h=6)) > score_transaction(_base_tx(velocity_24h=1))


def test_medium_velocity_adds_risk():
    assert score_transaction(_base_tx(velocity_24h=3)) > score_transaction(_base_tx(velocity_24h=1))


# ---------------------------------------------------------------------------
# score_transaction — failed logins
# ---------------------------------------------------------------------------

def test_failed_logins_below_2_adds_no_points():
    assert score_transaction(_base_tx(failed_logins_24h=1)) == 0


def test_failed_logins_2_adds_10():
    assert score_transaction(_base_tx(failed_logins_24h=2)) == 10


def test_failed_logins_mid_range_adds_10():
    assert score_transaction(_base_tx(failed_logins_24h=3)) == 10


def test_failed_logins_5_adds_20():
    assert score_transaction(_base_tx(failed_logins_24h=5)) == 20


def test_failed_logins_above_5_adds_20():
    assert score_transaction(_base_tx(failed_logins_24h=9)) == 20


def test_high_failed_logins_adds_risk():
    assert score_transaction(_base_tx(failed_logins_24h=5)) > score_transaction(_base_tx(failed_logins_24h=0))


def test_medium_failed_logins_adds_risk():
    assert score_transaction(_base_tx(failed_logins_24h=2)) > score_transaction(_base_tx(failed_logins_24h=0))


# ---------------------------------------------------------------------------
# score_transaction — prior chargebacks
# ---------------------------------------------------------------------------

def test_prior_chargebacks_0_adds_no_points():
    assert score_transaction(_base_tx(prior_chargebacks=0)) == 0


def test_prior_chargebacks_1_adds_5():
    assert score_transaction(_base_tx(prior_chargebacks=1)) == 5


def test_prior_chargebacks_2_adds_20():
    assert score_transaction(_base_tx(prior_chargebacks=2)) == 20


def test_prior_chargebacks_above_2_adds_20():
    assert score_transaction(_base_tx(prior_chargebacks=5)) == 20


def test_prior_chargebacks_2_adds_risk():
    assert score_transaction(_base_tx(prior_chargebacks=2)) > score_transaction(_base_tx(prior_chargebacks=0))


def test_prior_chargebacks_1_adds_risk():
    assert score_transaction(_base_tx(prior_chargebacks=1)) > score_transaction(_base_tx(prior_chargebacks=0))


def test_prior_chargebacks_2_scores_higher_than_1():
    assert score_transaction(_base_tx(prior_chargebacks=2)) > score_transaction(_base_tx(prior_chargebacks=1))


# ---------------------------------------------------------------------------
# score_transaction — clamping
# ---------------------------------------------------------------------------

def test_score_never_exceeds_100():
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=5,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100


def test_score_never_below_zero():
    assert score_transaction(_base_tx()) == 0


# ---------------------------------------------------------------------------
# composite — known fraud profile
# ---------------------------------------------------------------------------

def test_all_high_risk_signals_labels_high():
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=5,
        prior_chargebacks=2,
    )
    assert label_risk(score_transaction(tx)) == "high"
