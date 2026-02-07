# High-Dollar 10-Game Slate Strategy

Notes for adjusting strategy when playing high-entry-fee, 10-game slates (e.g. $360 contests). Based on contest analysis of DraftKings standings exports.

---

## Running Contest Analysis

Use [contest_analysis.py](contest_analysis.py) to analyze a contest CSV and generate a strategy report:

```bash
cd projection
python contest_analysis.py contests/$360NHLSpin_1.27.26.csv --out HIGH_DOLLAR_REPORT.md
```

**Options:**

- `--top-n 20,50,100` — Compute exposure and stack stats for top 20, 50, and 100 lineups (default: 20,50,100).
- `--out report.md` — Write the report to a Markdown file.
- `--salaries path/to/DKSalaries.csv` — Optional. If provided (or if a file exists in `daily_salaries/DKSalaries*.csv`), player names are mapped to teams so the report includes **stack size distribution** and **team concentration**. Without salaries, the report still includes **player exposure** and strategy implications.

**Output:** Contest metadata, player exposure in top-N lineups, stack size (primary/secondary) and team concentration when salaries are available, and a short "Strategy implications" section.

---

## Study a single pro

To study one DFS pro’s strategy (e.g. **bkreider**) and see what they do differently, filter the contest to that user’s entries and run the same metrics on their lineups only:

```bash
python contest_analysis.py contests/$360NHLSpin_1.27.26.csv --entry-name bkreider --out BKRIDER_PRO_REPORT.md
```

Pass the **base name** as shown in the CSV (e.g. `bkreider` matches entries like `bkreider (24/37)`). The report shows:

- **Pro study header** — Contest file, total contest entries, filter (EntryName = …), that pro’s entry count, rank range, and best score.
- **Player exposure** — % of that pro’s lineups containing each player (their chalk vs leverage).
- **Stack size** — Primary/secondary stack and distinct teams per lineup, averaged over the pro’s lineups.
- **Team concentration** — Which teams the pro used and how often (when run with `--salaries`).

Use this to see what the pro does differently (stack size, chalk, teams) and adjust your strategy—mirror their approach or deliberately differentiate.

---

## Recommended Strategy Adjustments (10-Game High-Dollar)

- **Primary stack size**: Use the contest report’s "Stack size (top N)" section. If top-20 lineups show primary stack avg ~4–5 players, target similar (current [config.py](config.py): `PREFERRED_PRIMARY_STACK_SIZE = 4`, `GPP_MAX_FROM_TEAM = 6`). For flatter slates you may see slightly smaller primary stacks; for concentrated slates, 5–6 from one team can be optimal.
- **Game concentration**: Use "Distinct teams per lineup (avg)" in the report. If top lineups average ~4 distinct teams (~2 games), concentrate in 2–3 games rather than spreading across all 10. Align with [DAILY_PROJECTION_IMPROVEMENT_PLAN.md](DAILY_PROJECTION_IMPROVEMENT_PLAN.md) ceiling-game identification (Vegas rank vs actual stack rank).
- **Chalk vs leverage**: The "Player exposure (top N)" table shows who actually showed up in winning lineups. Combine with your ownership model: high exposure + high ownership = chalk that hit; high exposure + low ownership = leverage that hit. Use the report to refine leverage targets for similar slates.
- **Team concentration**: When run with `--salaries`, the "Team concentration" table shows which teams appeared most in top lineups. Prioritize stacks from those teams (or their opponents for bring-backs) when slate structure is similar.

---

## Relation to Existing Config

Current GPP defaults in [config.py](config.py) (`GPP_MIN_STACK_SIZE`, `GPP_MAX_FROM_TEAM`, `PREFERRED_PRIMARY_STACK_SIZE`, `PREFERRED_SECONDARY_STACK_SIZE`) were tuned from earlier contest analysis (e.g. 1/22/26 slate). For high-dollar 10-game slates:

1. Run contest analysis on a recent $360 (or similar) contest and review the report.
2. If stack size or team concentration in winners differs from current defaults, consider a **one-off override** when building lineups (e.g. higher `max_from_team` or targeting 2–3 games only) rather than changing config globally.
3. Optionally add a **contest-type profile** (e.g. `high_dollar_10game`) in config and in the optimizer to apply different stack/team limits when you select that profile.

---

## Contest ROI / Leverage (Leverage Band + EV Ranking)

For **contest-specific** lineup selection (maximize long-term ROI by expected payout), use the contest ROI tool:

1. **Leverage recommendation** — Pass contest parameters to get a target ownership band and leverage tier:
   ```bash
   python contest_roi.py --entry-fee 360 --max-entries 1 --field-size 5000 --payout high_dollar_single
   ```
2. **Lineups ranked by EV** — In [main.py](main.py), pass contest params so lineups are scored by **contest expected value** and re-ordered by EV:
   ```bash
   python main.py --stacks --lineups 5 --contest-entry-fee 360 --contest-max-entries 1 --contest-field-size 5000 --contest-payout high_dollar_single
   ```
   The run prints a leverage recommendation, then generates lineups and re-ranks them by Contest EV ($). Use the top lineup(s) by EV for that contest.

Payout presets live in [config.py](config.py) (`CONTEST_PAYOUT_PRESETS`) and are used by [contest_roi.py](contest_roi.py). Full workflow: [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md) → "Contest-Specific Workflow (Leverage & EV)".

---

## Files

| File | Purpose |
|------|---------|
| [contest_analysis.py](contest_analysis.py) | Parse contest CSV, compute exposure/stacks/team concentration, emit Markdown report. |
| [contest_roi.py](contest_roi.py) | Contest profile, leverage recommendation, contest EV scorer; rank lineups by expected payout. |
| [contests/$360NHLSpin_1.27.26.csv](contests/$360NHLSpin_1.27.26.csv) | Example high-dollar contest standings export. |
| [config.py](config.py) | GPP stack/team limits; `CONTEST_PAYOUT_PRESETS` for contest ROI; override or add profile for high-dollar slates as needed. |
