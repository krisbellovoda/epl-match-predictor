from backend.app.model.config import (
    HALF_LIFE_DAYS,
    MODEL_VERSION,
    PRIOR_MATCHES,
    RECENT_FORM_ADJUSTMENT_ENABLED,
    get_model_configuration,
)


def test_model_has_version():
    assert MODEL_VERSION == "1.0.0"


def test_approved_team_strength_settings():
    assert PRIOR_MATCHES == 5.0
    assert HALF_LIFE_DAYS == 365.0


def test_experimental_recent_form_is_disabled():
    assert RECENT_FORM_ADJUSTMENT_ENABLED is False


def test_public_configuration_matches_constants():
    configuration = get_model_configuration()

    assert configuration["version"] == MODEL_VERSION

    assert (
        configuration["settings"]["prior_matches"]
        == PRIOR_MATCHES
    )

    assert (
        configuration["settings"]["half_life_days"]
        == HALF_LIFE_DAYS
    )