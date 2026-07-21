import { useState } from "react";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

function percentage(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function signedPercentage(value) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${percentage(value)}`;
}

function americanToDecimal(americanOdds) {
  if (
    americanOdds > -100 &&
    americanOdds < 100
  ) {
    throw new Error(
      "American odds must be +100 or higher, or -100 or lower.",
    );
  }

  if (americanOdds > 0) {
    return 1 + americanOdds / 100;
  }

  return 1 + 100 / Math.abs(americanOdds);
}

function MarketComparison({ prediction }) {
  const [oddsFormat, setOddsFormat] =
    useState("american");

  const [overOdds, setOverOdds] =
    useState("-110");

  const [underOdds, setUnderOdds] =
    useState("-110");

  const [comparison, setComparison] =
    useState(null);

  const [convertedOdds, setConvertedOdds] =
    useState(null);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function changeOddsFormat(event) {
    const newFormat = event.target.value;

    setOddsFormat(newFormat);
    setComparison(null);
    setConvertedOdds(null);
    setError("");

    if (newFormat === "american") {
      setOverOdds("-110");
      setUnderOdds("-110");
    } else {
      setOverOdds("1.91");
      setUnderOdds("1.91");
    }
  }

  async function compareMarket() {
    setError("");
    setComparison(null);
    setConvertedOdds(null);

    const parsedOverOdds = Number(overOdds);
    const parsedUnderOdds = Number(underOdds);

    if (
      !Number.isFinite(parsedOverOdds) ||
      !Number.isFinite(parsedUnderOdds)
    ) {
      setError("Enter valid odds for both outcomes.");
      return;
    }

    let decimalOverOdds;
    let decimalUnderOdds;

    try {
      if (oddsFormat === "american") {
        decimalOverOdds = americanToDecimal(
          parsedOverOdds,
        );

        decimalUnderOdds = americanToDecimal(
          parsedUnderOdds,
        );
      } else {
        if (
          parsedOverOdds <= 1 ||
          parsedUnderOdds <= 1
        ) {
          throw new Error(
            "Decimal odds must be greater than 1.",
          );
        }

        decimalOverOdds = parsedOverOdds;
        decimalUnderOdds = parsedUnderOdds;
      }
    } catch (conversionError) {
      setError(conversionError.message);
      return;
    }

    setConvertedOdds({
      over: decimalOverOdds,
      under: decimalUnderOdds,
    });

    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/market/compare`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_over_probability:
              prediction.total_goals.over_2_5,
            over_decimal_odds: decimalOverOdds,
            under_decimal_odds: decimalUnderOdds,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ?? "Market comparison failed.",
        );
      }

      setComparison(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const overValue =
    comparison?.expected_value.over_2_5 ?? 0;

  const underValue =
    comparison?.expected_value.under_2_5 ?? 0;

  const bestValue = Math.max(
    overValue,
    underValue,
  );

  const bestSide =
    overValue >= underValue
      ? "Over 2.5"
      : "Under 2.5";

  return (
    <section className="market-comparison">
      <h3>Check sportsbook odds</h3>

      <p className="market-description">
        Copy the Over 2.5 and Under 2.5 odds from your
        sportsbook. We will compare them with the model.
      </p>

      <label>
        Odds format
        <select
          value={oddsFormat}
          onChange={changeOddsFormat}
        >
          <option value="american">
            American odds
          </option>

          <option value="decimal">
            Decimal odds
          </option>
        </select>
      </label>

      <div className="odds-fields">
        <label>
          Over 2.5 odds
          <input
            type="number"
            step={
              oddsFormat === "american"
                ? "1"
                : "0.01"
            }
            value={overOdds}
            onChange={(event) =>
              setOverOdds(event.target.value)
            }
            placeholder={
              oddsFormat === "american"
                ? "-110"
                : "1.91"
            }
          />
        </label>

        <label>
          Under 2.5 odds
          <input
            type="number"
            step={
              oddsFormat === "american"
                ? "1"
                : "0.01"
            }
            value={underOdds}
            onChange={(event) =>
              setUnderOdds(event.target.value)
            }
            placeholder={
              oddsFormat === "american"
                ? "-110"
                : "1.91"
            }
          />
        </label>
      </div>

      <button
        type="button"
        onClick={compareMarket}
        disabled={loading}
      >
        {loading
          ? "Checking odds..."
          : "Check these odds"}
      </button>

      {error && <p className="error">{error}</p>}

      {convertedOdds &&
        oddsFormat === "american" && (
          <p className="converted-odds">
            Decimal equivalents: Over{" "}
            {convertedOdds.over.toFixed(2)} · Under{" "}
            {convertedOdds.under.toFixed(2)}
          </p>
        )}

      {comparison && (
        <div className="comparison-output">
          <div
            className={
              bestValue > 0
                ? "value-summary positive-summary"
                : "value-summary negative-summary"
            }
          >
            <span>Model conclusion</span>

            <strong>
              {bestValue > 0
                ? `Possible value: ${bestSide}`
                : "No positive value at these odds"}
            </strong>

            <p>
              {bestValue > 0
                ? `Estimated advantage at this price: ${signedPercentage(
                    bestValue,
                  )}`
                : "The available prices are not favorable based on this model."}
            </p>
          </div>

          <div className="simple-results">
            <article>
              <span>Sportsbook margin</span>

              <strong>
                {percentage(
                  comparison.sportsbook_margin,
                )}
              </strong>

              <small>
                The amount built into the sportsbook
                prices.
              </small>
            </article>

            <article>
              <span>Market estimate: Over 2.5</span>

              <strong>
                {percentage(
                  comparison.no_vig_market.over_2_5,
                )}
              </strong>

              <small>
                The market&apos;s estimated probability
                after removing its margin.
              </small>
            </article>

            <article>
              <span>Our estimate: Over 2.5</span>

              <strong>
                {percentage(
                  comparison.model.over_2_5,
                )}
              </strong>

              <small>
                The probability calculated by our model.
              </small>
            </article>

            <article>
              <span>Over 2.5 difference</span>

              <strong
                className={
                  comparison.edge.over_2_5 >= 0
                    ? "positive"
                    : "negative"
                }
              >
                {signedPercentage(
                  comparison.edge.over_2_5,
                )}
              </strong>

              <small>
                Positive means our model is more confident
                than the market.
              </small>
            </article>

            <article>
              <span>Value at Over 2.5 price</span>

              <strong
                className={
                  overValue >= 0
                    ? "positive"
                    : "negative"
                }
              >
                {signedPercentage(overValue)}
              </strong>

              <small>
                Positive is favorable. Negative means
                pass.
              </small>
            </article>

            <article>
              <span>Value at Under 2.5 price</span>

              <strong
                className={
                  underValue >= 0
                    ? "positive"
                    : "negative"
                }
              >
                {signedPercentage(underValue)}
              </strong>

              <small>
                Positive is favorable. Negative means
                pass.
              </small>
            </article>
          </div>
        </div>
      )}

      <p className="disclaimer">
        Model estimates are for educational analysis and
        do not guarantee profitable outcomes.
      </p>
    </section>
  );
}

export default MarketComparison;