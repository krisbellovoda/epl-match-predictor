MODEL_NAME = "EPL Team Strength Poisson Model"
MODEL_VERSION = "1.0.0"

# Approved production settings
PRIOR_MATCHES = 5.0
HALF_LIFE_DAYS = 365.0

# Experimental features that are not approved for production
RECENT_FORM_ADJUSTMENT_ENABLED = False
DIXON_COLES_ENABLED = False

# The over/under probability calibration passed its
# chronological evaluation and remains enabled.
PROBABILITY_CALIBRATION_ENABLED = True

TRAINING_LEAGUE = "English Premier League"
PRODUCTION_STATUS = "approved"


def get_model_configuration() -> dict:
    """
    Return public information about the production model.
    """

    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": PRODUCTION_STATUS,
        "league": TRAINING_LEAGUE,
        "method": (
            "Time-weighted attacking and defensive team "
            "strengths with independent Poisson score modeling"
        ),
        "settings": {
            "prior_matches": PRIOR_MATCHES,
            "half_life_days": HALF_LIFE_DAYS,
            "recent_form_adjustment_enabled": (
                RECENT_FORM_ADJUSTMENT_ENABLED
            ),
            "dixon_coles_enabled": (
                DIXON_COLES_ENABLED
            ),
            "probability_calibration_enabled": (
                PROBABILITY_CALIBRATION_ENABLED
            ),
        },
        "validation": {
            "selection_method": (
                "Chronological validation and holdout testing"
            ),
            "production_decision": (
                "Baseline retained after rolling-form, "
                "Dixon-Coles, and hybrid experiments failed "
                "to produce consistent holdout improvements"
            ),
        },
    }