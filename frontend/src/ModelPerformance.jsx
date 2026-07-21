function MetricCard({ label, value, detail, status = "neutral" }) {
  return (
    <article className={`performance-card ${status}`}>
      <span className="performance-label">{label}</span>
      <strong className="performance-value">{value}</strong>
      <p>{detail}</p>
    </article>
  );
}

function ModelPerformance() {
  return (
    <section className="model-performance">
      <div className="performance-heading">
        <div>
          <span className="section-eyebrow">MODEL EVALUATION</span>
          <h2>2025–26 holdout results</h2>
        </div>

        <span className="simulation-badge">
          Historical simulation
        </span>
      </div>

      <p className="performance-introduction">
        The model was trained using earlier matches and evaluated
        chronologically on the 2025–26 season. All results assume
        flat one-unit stakes and a minimum modeled edge of 5%.
      </p>

      <div className="performance-grid">
        <MetricCard
          label="Calibrated ROI"
          value="+4.37%"
          detail="Return per simulated unit staked"
          status="positive"
        />

        <MetricCard
          label="Simulated profit"
          value="+6.33 units"
          detail="Across 145 qualifying bets"
          status="positive"
        />

        <MetricCard
          label="Maximum drawdown"
          value="7.32 units"
          detail="Largest decline from a previous peak"
        />

        <MetricCard
          label="Raw model ROI"
          value="−4.19%"
          detail="Performance before probability calibration"
          status="negative"
        />
      </div>

      <div className="calibration-explanation">
        <h3>Why calibration mattered</h3>

        <p>
          The original Poisson model was overconfident. Logistic
          calibration moved its probabilities closer to observed
          outcomes, reducing the number of selected bets from 212
          to 145.
        </p>

        <p>
          These figures describe historical simulated performance.
          They do not guarantee that the model will produce future
          profits.
        </p>
      </div>
    </section>
  );
}

export default ModelPerformance;