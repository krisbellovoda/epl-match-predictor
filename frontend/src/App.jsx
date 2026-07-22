import { useEffect, useState } from "react";
import MarketComparison from "./MarketComparison";
import ModelPerformance from "./ModelPerformance";
import TeamSelect from "./TeamSelect";
import ScoreHeatmap from "./ScoreHeatmap";
import PredictionExplanation from "./PredictionExplanation";
import "./App.css";

const API_URL =
  "https://epl-match-predictor-api.onrender.com";


function formatPercentage(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function App() {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadTeams() {
      try {
        const response = await fetch(`${API_URL}/teams`);

        if (!response.ok) {
          throw new Error("Could not load teams.");
        }

        const data = await response.json();

        setTeams(data.teams);
        setHomeTeam(data.teams[0] ?? "");
        setAwayTeam(data.teams[1] ?? "");
      } catch (requestError) {
        setError(requestError.message);
      } finally {
        setLoadingTeams(false);
      }
    }

    loadTeams();
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();

    setError("");
    setPrediction(null);

    if (!homeTeam || !awayTeam) {
      setError("Choose a home and away team.");
      return;
    }

    if (homeTeam === awayTeam) {
      setError("Choose two different teams.");
      return;
    }

    setPredicting(true);

    try {
      const response = await fetch(
        `${API_URL}/predict/teams`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            home_team: homeTeam,
            away_team: awayTeam,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ?? "Prediction request failed.",
        );
      }

      setPrediction(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setPredicting(false);
    }
  }

  return (
    <main className="app">
      <header className="app-header">
  <div className="brand-row">
    <div
      className="app-mark"
      aria-hidden="true"
    >
      xG
    </div>

    <div className="brand-copy">
      <p className="eyebrow">
        Data-driven EPL analytics
      </p>

      <h1>EPL Match Predictor</h1>
    </div>
  </div>

  <p className="header-description">
    Select two teams to generate Poisson-based match
    probabilities from historical performance.
  </p>

  <span className="unofficial-label">
    Unofficial educational project
  </span>
</header>

      <section className="prediction-panel">
        <form onSubmit={handleSubmit}>
          <div className="team-fields">
            <TeamSelect
              label="Home team"
              teams={teams}
              value={homeTeam}
              onChange={setHomeTeam}
              disabled={loadingTeams}
            />

            <TeamSelect
              label="Away team"
              teams={teams}
              value={awayTeam}
              onChange={setAwayTeam}
              disabled={loadingTeams}
            />
          </div>

          <button
            type="submit"
            disabled={loadingTeams || predicting}
          >
            {predicting
              ? "Calculating..."
              : "Generate prediction"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}
      </section>

      {prediction && (
        <section className="results">
          <h2>
            {prediction.home_team} vs{" "}
            {prediction.away_team}
          </h2>

          <div className="expected-goals">
            <p>
              Home expected goals
              <strong>
                {prediction.expected_goals.home.toFixed(2)}
              </strong>
            </p>

            <p>
              Away expected goals
              <strong>
                {prediction.expected_goals.away.toFixed(2)}
              </strong>
            </p>
          </div>

          <h3>Match result</h3>

          <div className="probability-grid">
            <article>
              <span>Home win</span>

              <strong>
                {formatPercentage(
                  prediction.match_result.home_win,
                )}
              </strong>
            </article>

            <article>
              <span>Draw</span>

              <strong>
                {formatPercentage(
                  prediction.match_result.draw,
                )}
              </strong>
            </article>

            <article>
              <span>Away win</span>

              <strong>
                {formatPercentage(
                  prediction.match_result.away_win,
                )}
              </strong>
            </article>
          </div>

          <h3>Goal markets</h3>

          <div className="probability-grid">
            <article>
              <span>Over 2.5</span>

              <strong>
                {formatPercentage(
                  prediction.total_goals.over_2_5,
                )}
              </strong>
            </article>

            <article>
              <span>Under 2.5</span>

              <strong>
                {formatPercentage(
                  prediction.total_goals.under_2_5,
                )}
              </strong>
            </article>

            <article>
              <span>Both teams score</span>

              <strong>
                {formatPercentage(
                  prediction.both_teams_to_score.yes,
                )}
              </strong>
            </article>
          </div>

          <h3>Most likely scorelines</h3>

<div className="scorelines">
  {prediction.top_scorelines.map((result) => (
    <article key={result.score}>
      <strong>{result.score}</strong>

      <span>
        {formatPercentage(result.probability)}
      </span>
    </article>
  ))}
</div>

<ScoreHeatmap prediction={prediction} />

<PredictionExplanation
  prediction={prediction}
  homeTeam={homeTeam}
  awayTeam={awayTeam}
/>

<MarketComparison prediction={prediction} />
        </section>
      )}

            <ModelPerformance />

      <footer className="site-footer">
        <span>A project by</span>
        <strong>Kristi Bellovoda</strong>
      </footer>
    </main>
  );
}

export default App;