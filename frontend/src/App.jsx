import { useEffect, useState } from "react";
import MarketComparison from "./MarketComparison";
import ModelPerformance from "./ModelPerformance";
import "./App.css";


const API_URL = "http://127.0.0.1:8000";

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

    if (homeTeam === awayTeam) {
      setError("Choose two different teams.");
      return;
    }

    setPredicting(true);

    try {
      const response = await fetch(`${API_URL}/predict/teams`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
        }),
      });

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
      <header>
        <p className="eyebrow">Data-driven EPL analysis</p>
        <h1>EPL Match Predictor</h1>
        <p>
          Select two teams to generate Poisson-based match
          probabilities from historical performance.
        </p>
      </header>

      <section className="prediction-panel">
        <form onSubmit={handleSubmit}>
          <div className="team-fields">
            <label>
              Home team
              <select
                value={homeTeam}
                onChange={(event) =>
                  setHomeTeam(event.target.value)
                }
                disabled={loadingTeams}
              >
                {teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Away team
              <select
                value={awayTeam}
                onChange={(event) =>
                  setAwayTeam(event.target.value)
                }
                disabled={loadingTeams}
              >
                {teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </label>
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

          <MarketComparison prediction={prediction} />
        </section>
      )}

      <ModelPerformance />
    </main>
  );
}

export default App;