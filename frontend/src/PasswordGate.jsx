import { useState } from "react";




const ACCESS_CODE = "0408";
const SESSION_KEY = "epl-match-predictor-access";


function PasswordGate({ children }) {
  const [hasAccess, setHasAccess] = useState(
    () => sessionStorage.getItem(SESSION_KEY) === "granted"
  );

  const [code, setCode] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(event) {
    event.preventDefault();

    if (code === ACCESS_CODE) {
      sessionStorage.setItem(SESSION_KEY, "granted");
      setHasAccess(true);
      setCode("");
      setError("");
      return;
    }

    setError("Incorrect access code. Please try again.");
  }

  function lockSite() {
    sessionStorage.removeItem(SESSION_KEY);
    setHasAccess(false);
    setCode("");
    setError("");
  }

  if (!hasAccess) {
    return (
      <main className="access-page">
        <section className="access-card">
          <div className="access-mark" aria-hidden="true">
            xG
          </div>

          <span className="access-eyebrow">
            EPL ANALYTICS
          </span>

          <h1>Match intelligence, explained.</h1>

          <p className="access-description">
            Enter the demonstration access code to explore the
            Poisson prediction model, score probabilities, model
            evaluation, and sportsbook comparison tools.
          </p>

          <form
            className="access-form"
            onSubmit={handleSubmit}
          >
            <label htmlFor="access-code">
              Demonstration access code
            </label>

            <input
              id="access-code"
              type="password"
              inputMode="numeric"
              autoComplete="off"
              placeholder="Enter access code"
              value={code}
              onChange={(event) => {
                setCode(event.target.value);
                setError("");
              }}
              aria-describedby={
                error ? "access-error" : undefined
              }
              autoFocus
            />

            {error && (
              <p
                id="access-error"
                className="access-error"
                role="alert"
              >
                {error}
              </p>
            )}

            <button type="submit">
              Enter dashboard
            </button>
          </form>

          <p className="access-note">
            Unofficial educational analytics project. Predictions
            do not guarantee future outcomes.
          </p>
        </section>

        <div
          className="access-orb access-orb-one"
          aria-hidden="true"
        />

        <div
          className="access-orb access-orb-two"
          aria-hidden="true"
        />
      </main>
    );
  }

  return (
    <>
      <button
        className="lock-site-button"
        type="button"
        onClick={lockSite}
      >
        Lock site
      </button>

      {children}
    </>
  );
}


export default PasswordGate;