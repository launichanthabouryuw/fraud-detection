from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
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


# --- label_risk ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_exact_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# --- amount ---

def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) >= 25


def test_medium_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=600)) >= 10


# --- device risk ---

def test_high_device_risk_adds_points():
    low = score_transaction(_base_tx(device_risk_score=10))
    high = score_transaction(_base_tx(device_risk_score=75))
    assert high > low


def test_device_risk_boundary_70():
    score = score_transaction(_base_tx(device_risk_score=70))
    baseline = score_transaction(_base_tx(device_risk_score=10))
    assert score > baseline


def test_device_risk_boundary_40():
    score = score_transaction(_base_tx(device_risk_score=40))
    baseline = score_transaction(_base_tx(device_risk_score=10))
    assert score > baseline


# --- international ---

def test_international_adds_risk():
    domestic = score_transaction(_base_tx(is_international=0))
    intl = score_transaction(_base_tx(is_international=1))
    assert intl > domestic


# --- velocity ---

def test_high_velocity_adds_risk():
    low_v = score_transaction(_base_tx(velocity_24h=1))
    high_v = score_transaction(_base_tx(velocity_24h=6))
    assert high_v > low_v


def test_velocity_boundary_6():
    score = score_transaction(_base_tx(velocity_24h=6))
    baseline = score_transaction(_base_tx(velocity_24h=1))
    assert score > baseline


def test_medium_velocity_adds_risk():
    low_v = score_transaction(_base_tx(velocity_24h=1))
    med_v = score_transaction(_base_tx(velocity_24h=3))
    assert med_v > low_v


# --- prior chargebacks ---

def test_prior_chargebacks_2_adds_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    repeat = score_transaction(_base_tx(prior_chargebacks=2))
    assert repeat > clean


def test_prior_chargebacks_1_adds_risk():
    clean = score_transaction(_base_tx(prior_chargebacks=0))
    one = score_transaction(_base_tx(prior_chargebacks=1))
    assert one > clean


def test_prior_chargebacks_2_scores_higher_than_1():
    one = score_transaction(_base_tx(prior_chargebacks=1))
    two = score_transaction(_base_tx(prior_chargebacks=2))
    assert two > one


# --- failed logins ---

def test_high_failed_logins_adds_risk():
    clean = score_transaction(_base_tx(failed_logins_24h=0))
    high = score_transaction(_base_tx(failed_logins_24h=5))
    assert high > clean


# --- clamping ---

def test_score_never_exceeds_100():
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=5,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) <= 100


def test_score_never_below_zero():
    assert score_transaction(_base_tx()) >= 0


# --- composite high-risk case ---

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
