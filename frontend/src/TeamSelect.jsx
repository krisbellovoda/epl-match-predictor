import { useEffect, useRef, useState } from "react";

const TEAM_CODES = {
  Arsenal: "ARS",
  "Aston Villa": "AVL",
  Bournemouth: "BOU",
  Brentford: "BRE",
  Brighton: "BHA",
  Burnley: "BUR",
  Chelsea: "CHE",
  "Crystal Palace": "CRY",
  Everton: "EVE",
  Fulham: "FUL",
  Ipswich: "IPS",
  Leeds: "LEE",
  Leicester: "LEI",
  Liverpool: "LIV",
  Luton: "LUT",
  "Man City": "MCI",
  "Man United": "MUN",
  Newcastle: "NEW",
  "Nott'm Forest": "NFO",
  "Sheffield United": "SHU",
  Southampton: "SOU",
  Sunderland: "SUN",
  Tottenham: "TOT",
  "West Ham": "WHU",
  Wolves: "WOL",
};

function TeamBadge({ team }) {
  const abbreviation =
    TEAM_CODES[team] ??
    team
      .split(" ")
      .map((word) => word[0])
      .join("")
      .slice(0, 3)
      .toUpperCase();

  return (
    <span
      className="team-badge"
      aria-hidden="true"
    >
      {abbreviation}
    </span>
  );
}

function TeamSelect({
  label,
  teams,
  value,
  onChange,
  disabled = false,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    function handleOutsideClick(event) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener(
      "mousedown",
      handleOutsideClick,
    );

    return () => {
      document.removeEventListener(
        "mousedown",
        handleOutsideClick,
      );
    };
  }, []);

  function selectTeam(team) {
    onChange(team);
    setIsOpen(false);
  }

  return (
    <div
      className="team-select-field"
      ref={dropdownRef}
    >
      <span className="team-select-label">
        {label}
      </span>

      <button
        className="team-select-button"
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <span className="team-select-choice">
          <span className="team-name">
            {value || "Select a team"}
          </span>

          {value && <TeamBadge team={value} />}
        </span>

        <span
          className={`dropdown-chevron ${
            isOpen ? "open" : ""
          }`}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      {isOpen && !disabled && (
        <div
          className="team-options"
          role="listbox"
          aria-label={label}
        >
          {teams.map((team) => (
            <button
              className={`team-option ${
                team === value ? "selected" : ""
              }`}
              type="button"
              role="option"
              aria-selected={team === value}
              key={team}
              onClick={() => selectTeam(team)}
            >
              <span className="team-name">
                {team}
              </span>

              <TeamBadge team={team} />

              {team === value && (
                <span
                  className="selected-check"
                  aria-hidden="true"
                >
                  ✓
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default TeamSelect;