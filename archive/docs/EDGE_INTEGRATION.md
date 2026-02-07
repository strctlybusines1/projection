# NHL DFS Edge Integration (Tested & Working)

## Files to Install

```bash
mv ~/Downloads/edge_cache.py ~/Desktop/Code/projection/edge_cache.py
mv ~/Downloads/edge_stats.py ~/Desktop/Code/projection/edge_stats.py
mkdir -p ~/Desktop/Code/projection/cache
```

## What's Working

### Skater Edge (via NHL Stats API)
- Season summary stats (goals, assists, shots, etc.)
- Time on ice breakdown
- Cached daily

### Goalie Edge (NEW - via NHL Stats API)
- **EV Save %** - Even-strength save percentage (most predictive)
- **Quality Starts %** - % of games that are quality starts
- **PP/SH Save %** - Power play and shorthanded save %
- Cached daily

### Goalie Boost Factors

| Metric | Threshold | Boost |
|--------|-----------|-------|
| Elite EV SV% | ≥92.0% | +8% |
| Above-avg EV SV% | ≥90.5% | +4% |
| Poor EV SV% | <89.0% | -6% |
| Elite QS% | ≥60% | +6% |
| Above-avg QS% | ≥50% | +3% |
| Poor QS% | <40% | -4% |

### Real Results (from today's data)

| Goalie | Team | EV SV% | QS% | Boost |
|--------|------|--------|-----|-------|
| Jeremy Swayman | BOS | 92.3% | 68% | +14.5% |
| Logan Thompson | WSH | 92.1% | 63% | +14.5% |
| Andrei Vasilevskiy | TBL | 92.0% | 64% | +14.5% |
| Spencer Knight | CHI | 91.0% | 50% | +7.1% |
| Sergei Bobrovsky | FLA | 88.0% | 34% | -9.8% |

## Integration in main.py

Add after merging goalie projections (around line 920):

```python
# Apply Goalie Edge boosts
if args.edge and not args.no_edge:
    from edge_stats import apply_goalie_edge_boosts
    print("\nApplying goalie Edge stats...")
    goalies_merged = apply_goalie_edge_boosts(
        goalies_merged, 
        force_refresh=args.refresh_edge
    )
```

## Daily Workflow

```bash
# Morning (first run) - fetch fresh data
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# Rest of day - uses cache (instant)
python main.py --stacks --show-injuries --lineups 5 --edge
```

## Cache Files Created

```
projection/cache/
├── edge_stats_2026-02-04.json        # Skater data (864 players)
├── goalie_edge_stats_2026-02-04.json # Goalie data (92 goalies)
└── edge_metadata.json                 # Last fetch timestamps
```

## Performance

| Operation | Time |
|-----------|------|
| First fetch (both) | ~5 seconds |
| Cached load | <0.1 seconds |
| Old workflow | 3+ minutes per run |

---

**Note:** High-danger save % is NOT available in the NHL API. We use EV Save % and Quality Starts % instead, which are the next best predictive metrics according to the research papers.
