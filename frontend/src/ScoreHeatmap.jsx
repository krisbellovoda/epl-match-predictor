function formatProbability(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function ScoreHeatmap({ prediction }) {
  const scoreMatrix =
    prediction.score_matrix?.probabilities;

  if (!scoreMatrix) {
    return null;
  }

  const largestProbability = Math.max(
    ...scoreMatrix.flat(),
  );

  const goalLabels = scoreMatrix.map(
    (_, goalIndex) => goalIndex,
  );

  return (
    <section className="heatmap-section">
      <div className="heatmap-heading">
        <div>
          <h3>Score probability map</h3>

          <p>
            Darker cells represent more likely final scores.
          </p>
        </div>

        <div className="heatmap-legend">
          <span>Less likely</span>
          <i />
          <span>More likely</span>
        </div>
      </div>

      <div className="heatmap-scroll">
        <div
          className="score-heatmap"
          style={{
            gridTemplateColumns: `54px repeat(${goalLabels.length}, 1fr)`,
          }}
        >
          <div className="heatmap-corner">
            <span>Away →</span>
            <span>Home ↓</span>
          </div>

          {goalLabels.map((awayGoals) => (
            <div
              className="heatmap-axis-label"
              key={`away-${awayGoals}`}
            >
              {awayGoals}
            </div>
          ))}

          {scoreMatrix.map(
            (homeGoalProbabilities, homeGoals) => (
              <div
                className="heatmap-row"
                key={`home-${homeGoals}`}
                style={{ display: "contents" }}
              >
                <div className="heatmap-axis-label">
                  {homeGoals}
                </div>

                {homeGoalProbabilities.map(
                  (probability, awayGoals) => {
                    const intensity =
                      probability /
                      largestProbability;

                    const backgroundOpacity =
                      0.08 + intensity * 0.72;

                    const isMostLikely =
                      probability ===
                      largestProbability;

                    return (
                      <div
                        className={`heatmap-cell ${
                          isMostLikely
                            ? "most-likely"
                            : ""
                        }`}
                        key={`${homeGoals}-${awayGoals}`}
                        style={{
                          backgroundColor: `rgba(
                            14,
                            165,
                            233,
                            ${backgroundOpacity}
                          )`,
                        }}
                        title={
                          `${prediction.home_team} ` +
                          `${homeGoals}-${awayGoals} ` +
                          `${prediction.away_team}: ` +
                          formatProbability(probability)
                        }
                      >
                        <strong>
                          {homeGoals}-{awayGoals}
                        </strong>

                        <span>
                          {formatProbability(probability)}
                        </span>
                      </div>
                    );
                  },
                )}
              </div>
            ),
          )}
        </div>
      </div>
    </section>
  );
}

export default ScoreHeatmap;