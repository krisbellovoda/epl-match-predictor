MODEL_NAME = "EPL Match Prediction Model"
MODEL_VERSION = "1.1.0"

CURRENT_SEASON = "2026-27"

# Names match the names used by the model and frontend.
CURRENT_EPL_TEAMS = (
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich",
    "Leeds",
    "Liverpool",
    "Man City",
    "Man United",
    "Newcastle",
    "Nott'm Forest",
    "Sunderland",
    "Tottenham",
)

PROMOTED_TEAMS_2026_27 = (
    "Coventry City",
    "Hull City",
    "Ipswich",
)

# Coventry and Hull have no recent EPL observations in the
# training dataset. These conservative priors represent a
# below-average attack and above-average goals-conceded rate.
MISSING_PROMOTED_TEAM_PRIORS = {
    "Coventry City": {
        "attack": 0.85,
        "defense": 1.15,
    },
    "Hull City": {
        "attack": 0.85,
        "defense": 1.15,
    },
}

# Approved production settings
PRIOR_MATCHES = 5.0
HALF_LIFE_DAYS = 365.0

RECENT_FORM_ADJUSTMENT_ENABLED = False
DIXON_COLES_ENABLED = False
PROBABILITY_CALIBRATION_ENABLED = True

TRAINING_LEAGUE = "EPL"
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
        "season": CURRENT_SEASON,
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
            "dixon_coles_enabled": DIXON_COLES_ENABLED,
            "probability_calibration_enabled": (
                PROBABILITY_CALIBRATION_ENABLED
            ),
        },
        "roster": {
            "teams": list(CURRENT_EPL_TEAMS),
            "team_count": len(CURRENT_EPL_TEAMS),
            "promoted_teams": list(
                PROMOTED_TEAMS_2026_27
            ),
            "prior_based_teams": list(
                MISSING_PROMOTED_TEAM_PRIORS
            ),
        },
        "limitations": {
            "promoted_team_prior": (
                "Coventry City and Hull City use conservative "
                "promoted-club priors because the training data "
                "contains no recent EPL matches for those clubs."
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