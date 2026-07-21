function formatPercentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function PredictionExplanation({ prediction, homeTeam, awayTeam }) {
  if (!prediction) {
    return null;
  }

  const homeXg = prediction.expected_goals.home;
  const awayXg = prediction.expected_goals.away;
  const totalXg = homeXg + awayXg;

  const homeWin = prediction.match_result.home_win;
  const draw = prediction.match_result.draw;
  const awayWin = prediction.match_result.away_win;

  const over25 = prediction.total_goals.over_2_5;
  const btts = prediction.both_teams_to_score.yes;

  const resultOptions = [
    {
      label: `${homeTeam} win`,
      probability: homeWin,
    },
    {
      label: "Draw",
      probability: draw,
    },
    {
      label: `${awayTeam} win`,
      probability: awayWin,
    },
  ];

  const mostLikelyResult = resultOptions.reduce((best, current) =>
    current.probability > best.probability ? current : best
  );

  let expectedGameStyle;

  if (totalXg >= 3) {
    expectedGameStyle =
      "The model expects a relatively high-scoring match.";
  } else if (totalXg >= 2.3) {
    expectedGameStyle =
      "The model expects a moderate number of goals.";
  } else {
    expectedGameStyle =
      "The model expects a relatively low-scoring match.";
  }

  let teamAdvantage;

  if (Math.abs(homeXg - awayXg) < 0.2) {
    teamAdvantage =
      "The teams have similar expected-goal estimates, suggesting a competitive matchup.";
  } else if (homeXg > awayXg) {
    teamAdvantage =
      `${homeTeam} has the stronger attacking expectation in this matchup.`;
  } else {
    teamAdvantage =
      `${awayTeam} has the stronger attacking expectation in this matchup.`;
  }

  return (
    <section className="explanation-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">MODEL EXPLANATION</span>
          <h2>Why this prediction?</h2>
        </div>

        <span className="explanation-badge">
          Based on expected goals
        </span>
      </div>

      <div className="explanation-summary">
        <div className="explanation-icon" aria-hidden="true">
          i
        </div>

        <div>
          <span>Most likely result</span>

          <strong>
            {mostLikelyResult.label} at{" "}
            {formatPercentage(mostLikelyResult.probability)}
          </strong>
        </div>
      </div>

      <div className="explanation-grid">
        <article className="explanation-card">
          <span className="explanation-number">01</span>
          <h3>Team advantage</h3>
          <p>{teamAdvantage}</p>

          <div className="explanation-stat">
            <span>{homeTeam}</span>
            <strong>{homeXg.toFixed(2)} xG</strong>
          </div>

          <div className="explanation-stat">
            <span>{awayTeam}</span>
            <strong>{awayXg.toFixed(2)} xG</strong>
          </div>
        </article>

        <article className="explanation-card">
          <span className="explanation-number">02</span>
          <h3>Expected match style</h3>
          <p>{expectedGameStyle}</p>

          <div className="explanation-stat">
            <span>Combined expected goals</span>
            <strong>{totalXg.toFixed(2)}</strong>
          </div>

          <div className="explanation-stat">
            <span>Over 2.5 probability</span>
            <strong>{formatPercentage(over25)}</strong>
          </div>
        </article>

        <article className="explanation-card">
          <span className="explanation-number">03</span>
          <h3>Both teams scoring</h3>

          <p>
            The model gives both teams a{" "}
            {formatPercentage(btts)} chance of scoring at least
            once.
          </p>

          <div className="probability-track">
            <div
              className="probability-fill"
              style={{ width: formatPercentage(btts) }}
            />
          </div>

          <div className="probability-labels">
            <span>Unlikely</span>
            <span>Likely</span>
          </div>
        </article>
      </div>

      <p className="explanation-disclaimer">
        These explanations summarize the model’s probabilities.
        They are not guarantees and do not account for last-minute
        injuries, lineups, weather, or red cards.
      </p>
    </section>
  );
}

export default PredictionExplanation;