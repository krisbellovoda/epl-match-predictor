import { useEffect, useState } from "react";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

function formatPercentage(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatSignedPercentage(value) {
  const percentage = value * 100;
  const sign = percentage > 0 ? "+" : "";

  return `${sign}${percentage.toFixed(2)}%`;
}

function formatSignedUnits(value) {
  const sign = value > 0 ? "+" : "";

  return `${sign}${value.toFixed(2)} units`;
}

function MetricCard({
  label,
  value,
  detail,
  status = "neutral",
}) {
  return (
    <article className={`performance-card ${status}`}>
      <span className="performance-label">{label}</span>
      <strong className="performance-value">{value}</strong>
      <p>{detail}</p>
    </article>
  );
}

function ModelPerformance() {
  const [performance, setPerformance] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadPerformance() {
      try {
        const response = await fetch(
          `${API_URL}/model/performance`,
        );

        if (!response.ok) {
          throw new Error(
            "The performance request failed.",
          );
        }

        const data = await response.json();
        setPerformance(data);
      } catch {
        setError(
          "Model performance is temporarily unavailable.",
        );
      }
    }

    loadPerformance();
  }, []);

  if (error) {
    return (
      <section className="model-performance">
        <span className="section-eyebrow">
          MODEL EVALUATION
        </span>

        <p className="performance-error">{error}</p>
      </section>
    );
  }

  if (!performance) {
    return (
      <section className="model-performance">
        <span className="section-eyebrow">
          MODEL EVALUATION
        </span>

        <p className="performance-loading">
          Loading evaluation results...
        </p>
      </section>
    );
  }

  return (
    <section className="model-performance">
      <div className="performance-heading">
        <div>
          <span className="section-eyebrow">
            MODEL EVALUATION
          </span>

          <h2>
            {performance.test_season.replace("_", "–")}{" "}
            holdout results
          </h2>
        </div>

        <span className="simulation-badge">
          Historical simulation
        </span>
      </div>

      <p className="performance-introduction">
        The model was trained using earlier matches and
        evaluated chronologically on the selected season.
        Results use flat one-unit stakes and a minimum modeled
        edge of{" "}
        {formatPercentage(performance.minimum_edge)}.
      </p>

      <div className="performance-grid">
        <MetricCard
          label="Calibrated ROI"
          value={formatSignedPercentage(performance.roi)}
          detail="Return per simulated unit staked"
          status="positive"
        />

        <MetricCard
          label="Simulated profit"
          value={formatSignedUnits(
            performance.profit_units,
          )}
          detail={`Across ${performance.bets} qualifying bets`}
          status="positive"
        />

        <MetricCard
          label="Maximum drawdown"
          value={`${performance.maximum_drawdown.toFixed(
            2,
          )} units`}
          detail="Largest decline from a previous peak"
        />

        <MetricCard
          label="Raw model ROI"
          value={formatSignedPercentage(
            performance.raw_model_roi,
          )}
          detail="Performance before probability calibration"
          status="negative"
        />
      </div>

      <div className="calibration-explanation">
        <h3>Why calibration mattered</h3>

        <p>
          The original Poisson model was overconfident.
          Logistic calibration moved its probabilities closer
          to observed outcomes and made bet selection more
          conservative.
        </p>

        <p>
          The calibrated model recorded{" "}
          {performance.wins} wins and {performance.losses}{" "}
          losses, with a{" "}
          {formatPercentage(performance.hit_rate)} hit rate at
          average decimal odds of{" "}
          {performance.average_odds.toFixed(2)}.
        </p>

        <p>
          These figures describe historical simulated
          performance and do not guarantee future profits.
        </p>
      </div>
    </section>
  );
}

export default ModelPerformance;