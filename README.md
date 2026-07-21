# EPL Match Predictor

A full-stack EPL ANALYTICS platform that generates match probabilities using time-weighted team strengths, independent Poisson score modeling, probability calibration, and chronological backtesting.

The project includes a FastAPI backend, React dashboard, sportsbook comparison tools, score-probability visualization, explainable predictions, versioned model artifacts, and a documented model-validation process.

> This is an unofficial educational project. It is not affiliated with the Premier League, its clubs, or any sportsbook.

## Live demonstration

The public demonstration uses a simple client-side access gate.

**Demonstration access code: `0408`**

This code is intentionally public and is not real authentication. It only acts as a presentation gate for the portfolio demonstration.

Live URLs will be added after deployment:

- Frontend: `Coming soon`
- API: `Coming soon`
- API documentation: `Coming soon`

## Project overview

The application allows a user to:

- Select a home and away team.
- Generate expected-goal estimates.
- View home-win, draw, and away-win probabilities.
- Evaluate over/under 2.5 goals.
- Evaluate both-teams-to-score probabilities.
- Explore the most likely scorelines.
- Inspect a complete score-probability heatmap.
- Read a plain-language explanation of the prediction.
- Enter sportsbook odds in American or decimal format.
- Remove the sportsbook margin and compare market probabilities.
- Calculate model edge and expected value.
- Review historical holdout and betting-simulation performance.
- Inspect the production model version and approved settings.

## Technology stack

### Backend

- Python 3
- FastAPI
- Pydantic
- Pandas
- NumPy
- SciPy
- Uvicorn
- Pytest

### Frontend

- React
- Vite
- JavaScript
- CSS
- ESLint

### Development and deployment

- Git
- GitHub
- Environment-based API configuration
- Environment-based CORS configuration
- Versioned production model artifacts

## Architecture

```text
React frontend
      |
      | HTTP/JSON
      v
FastAPI backend
      |
      +-- Team-strength model
      +-- Poisson score model
      +-- Probability calibrator
      +-- Market comparison utilities
      +-- Versioned production artifact
```

The deployed backend does not require the raw historical CSV files. The approved team-strength model is trained locally and exported to:

```text
backend/app/model/artifacts/team_strength_model.json
```

The API loads this artifact when it starts.

## Model methodology

### 1. Team attacking and defensive strength

The production model estimates separate team ratings for:

- Home attacking strength
- Home defensive strength
- Away attacking strength
- Away defensive strength

Each rating is measured relative to the appropriate league-average scoring rate.

For example, the home expected-goal estimate is conceptually:

```text
league home scoring rate
× home team's home attacking strength
× away team's away defensive weakness
```

### 2. Small-sample shrinkage

Teams with limited observations are pulled toward league-average performance.

The approved production setting is:

```text
prior_matches = 5
```

This reduces extreme ratings caused by small samples.

### 3. Time weighting

Older matches receive less weight than recent matches.

The approved production setting is:

```text
half_life_days = 365
```

A match approximately 365 days old receives half the weight of a current match.

### 4. Poisson score model

Home and away goals are modeled using Poisson distributions.

The model constructs a score matrix containing the probability of each result:

```text
0-0, 0-1, 0-2, ...
1-0, 1-1, 1-2, ...
2-0, 2-1, 2-2, ...
```

The score matrix is used to calculate:

- Match-result probabilities
- Over/under 2.5 probabilities
- Both-teams-to-score probabilities
- Most likely scorelines
- Score-probability heatmap

### 5. Probability calibration

The original over/under model was overconfident.

A logistic probability calibrator was trained using an earlier season and evaluated on a later holdout season. Calibration moved predicted probabilities closer to observed outcome frequencies.

The production API applies this calibrator to the over/under market.

## Production model

```text
Model name: EPL Team Strength Poisson Model
Model version: 1.0.0
Status: approved
Prior matches: 5
Half-life: 365 days
Recent-form adjustment: disabled
Dixon-Coles adjustment: disabled
Probability calibration: enabled
```

The configuration is centralized in:

```text
backend/app/model/config.py
```

Public model information is available through:

```text
GET /model/info
```

## Historical data

The development dataset contains:

```text
Seasons: 3
Matches used to produce the current artifact: 1,140
Teams represented: 25
```

The project uses historical English football match data, including:

- Match date
- Home and away teams
- Full-time goals
- Match result
- Shots
- Shots on target
- Historical sportsbook odds

Raw data files remain local and are excluded from Git. This keeps model training separate from production inference and avoids making the deployed API depend on local CSV files.

When using or redistributing football data, review and follow the original data provider's applicable terms and attribution requirements.

## Chronological evaluation

The project uses chronological evaluation rather than random train/test splitting.

```text
Past matches → training
Later matches → validation
Newest season → holdout test
```

This better reflects real prediction conditions and prevents future information from entering earlier predictions.

### Production baseline holdout

The approved `5 prior matches / 365-day half-life` model recorded approximately:

| Metric | Result |
|---|---:|
| Matches tested | 380 |
| Match-result accuracy | 48.42% |
| Over/under 2.5 accuracy | 53.68% |
| Multiclass Brier score | 0.6196 |
| Log loss | 1.0322 |
| Total-goals MAE | 1.2528 |

For Brier score, log loss, and MAE, lower values are better.

## Calibrated betting simulation

A historical holdout simulation compared calibrated model probabilities with closing sportsbook over/under odds.

The simulation used:

```text
Flat stake: 1 unit
Minimum modeled edge: 5 percentage points
Test season: 2025-26 holdout
```

### Calibrated results

| Metric | Result |
|---|---:|
| Qualifying bets | 145 |
| Wins | 72 |
| Losses | 73 |
| Hit rate | 49.66% |
| Simulated profit | +6.33 units |
| Simulated ROI | +4.37% |
| Average decimal odds | 2.13 |
| Maximum drawdown | 7.32 units |

### Raw-model comparison

Before calibration, the same general strategy produced approximately:

```text
Raw model ROI: -4.19%
```

This experiment demonstrated that probability quality can matter more than classification accuracy when probabilities are used for market decisions.

> Historical simulated returns do not guarantee future profitability. The simulation may not capture limits, line movement, rejected wagers, latency, market availability, or execution costs.

## Experiments that were not deployed

An important goal of the project was to evaluate ideas objectively rather than automatically adding complexity.

### Three-season strength history

Adding historical seasons improved some probability metrics compared with the initial one-season baseline. This motivated the use of time weighting and shrinkage.

### Dixon-Coles adjustment

A Dixon-Coles low-score correction was implemented and tested.

The holdout negative-log-likelihood improvement was extremely small. The adjustment was retained as documented research code but was not enabled in production.

### Rolling recent-form features

Leakage-safe rolling features were created for:

- Recent goals scored
- Recent goals conceded
- Recent points
- Venue-specific scoring
- Venue-specific defending
- Shots
- Shots on target

The feature-generation order was:

```text
Read prior history
→ create current pre-match features
→ observe current result
→ update history
```

This prevents the current match from affecting its own prediction.

A rolling-feature Poisson regression underperformed the production baseline:

| Metric | Production baseline | Rolling model |
|---|---:|---:|
| Match-result accuracy | 47.9–48.4% | 45.9% |
| Over 2.5 accuracy | 53.7–56.3% | 54.9% |
| Brier score | approximately 0.620 | 0.635 |
| Log loss | approximately 1.032 | 1.054 |

The rolling model was therefore rejected.

### Hybrid long-term strength and recent form

A hybrid model applied a capped short-term adjustment to the long-term team-strength expected goals.

Validation selected an adjustment strength of:

```text
0.20
```

On the final holdout it produced:

- A very small match-accuracy improvement
- A very small log-loss improvement
- Worse Brier score
- Worse over/under accuracy
- Worse total-goal error

The improvement was too inconsistent to justify deployment.

### Block-bootstrap uncertainty analysis

A paired moving-block bootstrap used 5,000 resamples with blocks of 10 chronological matches.

The 95% confidence intervals for accuracy, Brier score, and log-loss changes all crossed zero.

Therefore, the hybrid model's apparent improvement was statistically inconclusive.

### Hyperparameter tuning

The team-strength parameters were tuned on the `2024-25` validation season.

Validation favored:

```text
prior_matches = 3
half_life_days = 730
```

However, those settings underperformed the existing `5 / 365` configuration on the final `2025-26` holdout.

This was treated as validation overfitting, and the existing production configuration was retained.

## Why rejected experiments matter

The production model is not simply the most complicated model tested.

The development process included:

- Leakage-safe feature engineering
- Chronological validation
- Separate holdout testing
- Probability scoring
- Calibration analysis
- Hyperparameter tuning
- Bootstrap uncertainty estimation
- Explicit rejection of changes that failed to generalize

This creates a more defensible modeling process than selecting a model using accuracy from a single random split.

## User interface

The React dashboard includes:

- Custom team selectors
- Neutral team abbreviation badges
- Responsive layout
- Expected-goal display
- Match-result probability cards
- Goal-market probabilities
- Most-likely-score cards
- Interactive score heatmap
- Prediction explanation panel
- Sportsbook market comparison
- American and decimal odds support
- Calibration performance panel
- Model evaluation summary
- Demonstration access gate
- Mobile-responsive styling

## Sportsbook comparison

Users can enter over/under 2.5 odds in either:

- American format, such as `-110` or `+125`
- Decimal format, such as `1.91` or `2.25`

The application:

1. Converts odds into implied probabilities.
2. Calculates the sportsbook margin.
3. Removes the margin to estimate no-vig probabilities.
4. Compares the market with the calibrated model.
5. Reports model edge.
6. Reports expected value.

## API endpoints

The FastAPI backend exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Runtime and artifact health |
| GET | `/teams` | Available teams |
| GET | `/model/info` | Production model configuration |
| GET | `/model/performance` | Historical evaluation summary |
| POST | `/predict` | Prediction from manually entered xG |
| POST | `/predict/teams` | Prediction from selected teams |
| POST | `/market/compare` | Model-versus-market comparison |

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

when running locally.

## Project structure

```text
epl-match-predictor/
├── backend/
│   ├── app/
│   │   ├── data_loader.py
│   │   ├── main.py
│   │   └── model/
│   │       ├── artifacts/
│   │       │   ├── backtest_summary.json
│   │       │   ├── over_2_5_calibrator.json
│   │       │   └── team_strength_model.json
│   │       ├── backtest.py
│   │       ├── baseline_tuning.py
│   │       ├── betting_backtest.py
│   │       ├── bootstrap_comparison.py
│   │       ├── calibration.py
│   │       ├── calibration_service.py
│   │       ├── config.py
│   │       ├── dixon_coles.py
│   │       ├── hybrid_backtest.py
│   │       ├── market.py
│   │       ├── model_artifact.py
│   │       ├── poisson.py
│   │       ├── probability_calibrator.py
│   │       ├── rolling_features.py
│   │       ├── rolling_goal_model.py
│   │       └── team_strength.py
│   ├── data/
│   │   └── raw/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── tests/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── MarketComparison.jsx
│   │   ├── ModelPerformance.jsx
│   │   ├── PasswordGate.jsx
│   │   ├── PredictionExplanation.jsx
│   │   ├── ScoreHeatmap.jsx
│   │   ├── TeamSelect.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── .gitignore
└── README.md
```

The exact structure may evolve as additional features are introduced.

## Local installation

### Requirements

Install:

- Python 3.13 or a compatible recent Python version
- Node.js
- Git
- Visual Studio Code or another editor

### Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/epl-match-predictor.git
cd epl-match-predictor
```

### Create the Python environment

Windows PowerShell:

```powershell
py -m venv .venv
```

If PowerShell blocks activation, activation is optional. Use the virtual environment's Python directly.

### Install backend dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
```

### Run backend tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -v
```

### Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Install frontend dependencies

```powershell
npm.cmd --prefix frontend install
```

### Start the frontend

Open a second terminal:

```powershell
npm.cmd --prefix frontend run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

Use demonstration access code:

```text
0408
```

## Production builds

### Backend

Install only production dependencies:

```bash
pip install -r backend/requirements.txt
```

Example production command:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

### Frontend

```bash
cd frontend
npm install
npm run build
```

The frontend build output is:

```text
frontend/dist
```

## Environment variables

### Frontend

```text
VITE_API_URL=https://your-api-domain.example
```

`VITE_` variables are included in browser code and must not contain secrets.

### Backend

```text
FRONTEND_ORIGINS=https://your-frontend-domain.example
```

Multiple allowed frontend origins can be separated by commas:

```text
FRONTEND_ORIGINS=https://site-one.example,https://site-two.example
```

## Updating the production model

Raw CSV files are used locally for research and retraining. They are intentionally excluded from Git.

After updating the historical data and approving a new model configuration, regenerate the production artifact:

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.data_loader import load_match_data; from backend.app.model.model_artifact import save_production_model; artifact = save_production_model(load_match_data()); print({'version': artifact['model_version'], 'matches': artifact['training_matches'], 'teams': len(artifact['model']['teams'])})"
```

Then:

1. Run the complete test suite.
2. Inspect the updated artifact.
3. Increment the model version when appropriate.
4. Commit the new artifact.
5. Push the commit.
6. Allow the hosting platform to redeploy.

## Testing

Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests -v
```

Run frontend linting:

```powershell
npm.cmd --prefix frontend run lint
```

Create a frontend production build:

```powershell
npm.cmd --prefix frontend run build
```

At the time of this README update, the backend test suite contains:

```text
33 passing tests
```

Tests cover:

- Probability sums
- Expected-goal validation
- Score-matrix behavior
- Dixon-Coles behavior
- Market calculations
- Model configuration
- Rolling-feature leakage prevention
- Team-strength estimation
- Scoreline results

## Deployment workflow

The expected production workflow is:

```text
Edit locally
→ test locally
→ commit with Git
→ push to GitHub
→ hosting platforms redeploy
→ run live smoke tests
```

The frontend and backend are deployed separately:

```text
React frontend → static frontend host
FastAPI backend → Python web-service host
```

The frontend receives the deployed API URL through `VITE_API_URL`. The backend receives the deployed frontend URL through `FRONTEND_ORIGINS`.

## Known limitations

The model does not currently include reliable real-time information for:

- Confirmed starting lineups
- Injuries and suspensions
- Player-level expected minutes
- Weather
- Tactical changes
- Manager changes
- Live match events
- Red-card probability
- Fixture congestion beyond historical weighting
- Transfer-window squad changes
- Market liquidity
- Bet limits
- Execution delays

The Poisson model also assumes a simplified scoring process. Football scores may exhibit dependence and tactical effects that are not fully represented by independent Poisson distributions.

## Security note

The demonstration access code `0408` is implemented in frontend JavaScript and documented publicly here.

It is not secure authentication.

Real authentication would require:

- Backend identity verification
- Hashed passwords
- Secure sessions or tokens
- Rate limiting
- HTTPS
- Account and authorization controls

The access gate exists only to create a polished portfolio demonstration experience.

## Responsible-use disclaimer

This project is intended for:

- Education
- Statistical analysis
- Software-engineering demonstration
- Portfolio presentation

It does not provide financial advice and does not guarantee profitable outcomes.

Anyone using betting-related information is responsible for following applicable laws, age requirements, platform rules, and responsible-gambling practices.

Never wager money that you cannot afford to lose.

## Resume summary

Possible résumé description:

> Built a full-stack English football probability platform using React, FastAPI, Pandas, SciPy, and Poisson modeling. Engineered time-weighted attacking and defensive team ratings, logistic probability calibration, no-vig market comparison, chronological backtesting, versioned production artifacts, and a responsive analytics dashboard. Evaluated and rejected non-generalizing Dixon-Coles, rolling-form, hybrid, and hyperparameter-tuned variants using holdout scoring and moving-block bootstrap uncertainty analysis.

## Future improvements

Potential future work includes:

- Automated data-update pipeline
- Player and lineup features
- Injury and suspension data
- Opponent-adjusted shot-quality ratings
- Separate promoted-team priors
- Additional goal totals
- Asian handicap probabilities
- Prediction-history storage
- Model-monitoring dashboard
- Scheduled recalibration
- Stronger backend authentication
- Continuous integration
- Automated deployment checks

## Author

Created as a full-stack data-science and software-engineering portfolio project.



```text
Name: Kristi Bellovoda
GitHub: https://github.com/krisbellovoda
LinkedIn: www.linkedin.com/in/kris-bellovoda-7a2b35243

```