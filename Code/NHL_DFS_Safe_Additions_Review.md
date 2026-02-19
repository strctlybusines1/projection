# NHL DFS Projection System — Safe Additions Review

**Reviewed:** February 6, 2026  
**Codebase:** ~14,340 lines across 21 Python modules  
**Principle:** The core projection pipeline is profitable. Don't touch it. Add tools around the edges.

---

## System Assessment

You've built a serious, production-grade NHL DFS system. The pipeline — data ingestion → feature engineering → projections → ownership → optimization → output — is clean, well-documented, and most importantly, making money. The Edge stats integration, single-entry selector, linemate correlation analysis, and contest-aware EV scoring are features that most DFS tools don't have.

Everything below is **additive only**. None of these changes modify your projection math, bias corrections, Edge thresholds, feature engineering, or ownership model.

---

## 1. ILP Optimizer (Drop-In Replacement)

**What it does:** Takes your *exact same projections* and finds mathematically guaranteed optimal lineup combinations under the salary cap. Your current greedy optimizer is good but can miss better salary constructions.

**Why it's safe:** It sits *downstream* of your projections. The input is identical — projected FPTS, salary, position, team. Only the lineup assembly logic changes.

**What you gain:** Guaranteed optimal lineups, easy player lock/exclude, built-in exposure limits for multi-entry, and lineup diversity constraints between iterations.

**Tool:** PuLP (free, lightweight, `pip install pulp`)

**Risk to current model:** Zero.

---

## 2. Historical SQLite Database

**What it does:** Consolidates your scattered CSVs and xlsx files (daily_projections, contests, backtests, daily_salaries) into a single queryable database file.

**Why it's safe:** Purely organizational. Doesn't change any logic. Your existing files stay exactly where they are — this just gives you a structured way to query across slates.

**What you gain:** Run queries like "my ROI by contest type," "players I consistently over/under-project," "which stack patterns have historically cashed." Foundation for future analysis without touching the model.

**Tool:** SQLite (built into Python, zero infrastructure)

**Risk to current model:** Zero.

---

## 3. Basic pytest Suite

**What it does:** Adds automated tests that verify your core functions still produce expected outputs. Things like: "does `normalize_position('LW')` return `'W'`?", "does the salary cap constraint hold?", "does the ownership model return values between 0 and 100?"

**Why it's safe:** Tests only *observe* your code — they don't change it. They protect you from accidentally breaking something during future edits.

**What you gain:** Confidence when making any changes. Run `pytest` before any slate and know the system is healthy.

**Tool:** pytest (`pip install pytest`)

**Risk to current model:** Zero.

---

## 4. Shell Script Shortcuts

**What it does:** Wraps your most common command combos into simple shortcuts.

```bash
./run.sh se          # Single-entry: 40 candidates, edge, single-entry selector
./run.sh gpp 20      # GPP: 20 lineups with stacks and edge
./run.sh base        # Quick base projections, no edge
./run.sh backtest    # Post-slate backtest
```

**Why it's safe:** Just convenience aliases for commands you already run.

**What you gain:** Fewer typos, faster workflow, no forgetting flags.

**Risk to current model:** Zero.

---

## 5. Pre-Flight Validation Script

**What it does:** A `--validate` check that runs before lineup export and confirms:

- Salary file is from today (not stale)
- All lineup players exist in DK salary file
- Goalie is confirmed on DailyFaceoff
- 3-team minimum is met
- No injured/scratched players slipped through
- Salary is under $50,000

**Why it's safe:** Read-only checks on your existing output. Catches human error, doesn't change the model.

**What you gain:** Peace of mind before lock. Catches the "stale salary file" gotcha you already documented in CLAUDE.md.

**Risk to current model:** Zero.

---

## 6. ROI Tracking Dashboard

**What it does:** Adds a page to your existing Flask dashboard that shows cumulative ROI by contest type, win rate, average finish percentile, and best/worst plays over time.

**Why it's safe:** Pure visualization of results you already have in your contests/ folder. Doesn't feed back into the model.

**What you gain:** Clear picture of where you're making money (SE vs. multi-entry, small vs. large fields, which contest sizes). Helps you allocate bankroll more intelligently.

**Risk to current model:** Zero.

---

## 7. Documentation Consolidation

**What it does:** Merges your 15+ markdown files into 3 clean docs:

- **README.md** — Setup, architecture, module reference
- **WORKFLOW.md** — Daily operations (currently split across CLAUDE.md and DAILY_WORKFLOW.md which overlap)
- **RESEARCH.md** — Strategy notes, backtest case studies, improvement ideas

Archive the PRO_REPORT files into a `reports/` folder. Remove the NFLParser file from the root.

**Why it's safe:** No code changes at all.

**What you gain:** One place to look for any given piece of information instead of searching across 15 files.

**Risk to current model:** Zero.

---

## 8. Automated Post-Slate Backtest

**What it does:** After games finish (~midnight ET), automatically fetches actual scores from the NHL API and runs the backtest — no manual CSV creation needed.

**Why it's safe:** The backtest module already exists and works. This just automates the data collection step that you currently do manually.

**What you gain:** Consistent daily backtesting without the friction of manual data entry. More data points for tracking model accuracy over time.

**Tool:** A cron job or launchd script (you already have `run_daily.sh` and `logs/launchd_*.log` so you're familiar with this)

**Risk to current model:** Zero.

---

## 9. Git Remote Backup

**What it does:** Push your existing git repo to a private GitHub/GitLab repository.

**Why it's safe:** Just backup. Doesn't change any files.

**What you gain:** Version history, ability to revert any change, backup if your machine dies, and you can see exactly what changed and when if something breaks.

**Risk to current model:** Zero.

---

## 10. Rich CLI Output

**What it does:** Replaces plain `print()` statements with formatted tables, colored text, and progress bars using the Rich library.

**Why it's safe:** Cosmetic only — changes how output *looks*, not what it *computes*.

**What you gain:** Much easier to scan projection output, spot value plays, and read backtest results at a glance.

**Tool:** Rich (`pip install rich`)

**Risk to current model:** Zero.

---

## Priority Order

| Priority | Addition | Effort | Impact |
|----------|----------|--------|--------|
| 1 | ILP Optimizer | Medium | High — better lineups from same projections |
| 2 | Shell script shortcuts | Low | Medium — daily workflow speed |
| 3 | Pre-flight validation | Low | Medium — prevents costly mistakes |
| 4 | Git remote backup | Low | High — protects everything you've built |
| 5 | Historical database | Medium | High — foundation for analysis |
| 6 | pytest suite | Medium | Medium — safety net for future changes |
| 7 | Doc consolidation | Low | Low — quality of life |
| 8 | Auto post-slate backtest | Medium | Medium — more consistent tracking |
| 9 | ROI dashboard | Medium | Medium — bankroll intelligence |
| 10 | Rich CLI output | Low | Low — cosmetic |

---

## Goalie Model — Parked for Discussion

You mentioned you have an idea about the goalie model. That's separate from everything above — whenever you're ready to explain it, I'm all ears. We can evaluate whether it's a safe addition or something that needs careful A/B testing against your current approach.

---

*None of the above touches your projection math, bias corrections, Edge boosts, feature weights, ownership model, or stacking logic. The engine stays exactly as-is.*
