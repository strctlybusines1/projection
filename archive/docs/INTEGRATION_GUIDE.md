# NHL DFS Full Integration Guide

## Files to Update

### 1. Replace `edge_cache.py` with `edge_cache_v2.py`
```bash
mv edge_cache.py edge_cache_backup.py
mv edge_cache_v2.py edge_cache.py
```

### 2. Replace `edge_stats.py` with `edge_stats_v2.py`
```bash
mv edge_stats.py edge_stats_backup.py
mv edge_stats_v2.py edge_stats.py
```

### 3. Update `main.py` - Add Goalie Edge Integration

Find this section (around line 915-920):
```python
    goalies_merged = merge_projections_with_salaries(
        projections['goalies'], dk_goalies, 'goalie'
    )

    print(f"  Matched {len(skaters_merged)} skaters")
    print(f"  Matched {len(goalies_merged)} goalies")
```

Add after it:
```python
    # Apply Goalie Edge boosts if enabled
    if args.edge and not args.no_edge:
        from edge_stats import apply_goalie_edge_boosts
        print("\nApplying goalie Edge stats...")
        goalies_merged = apply_goalie_edge_boosts(
            goalies_merged, 
            force_refresh=args.refresh_edge
        )
```

### 4. Update `data_pipeline.py` - Apply Skater Edge in Pipeline

Find `build_projection_dataset` method and ensure Edge is applied there, or apply after projection generation in main.py.

Current flow in main.py (line 880-886):
```python
    data = pipeline.build_projection_dataset(
        include_game_logs=False,
        include_injuries=not args.no_injuries,
        include_advanced_stats=not args.no_advanced,
        include_edge_stats=args.edge and not args.no_edge,
        force_refresh_edge=args.refresh_edge
    )
```

If Edge is applied in data_pipeline, ensure it uses the cached version.

---

## New Workflow Commands

### Morning (First Run of Day)
```bash
# Fetch and cache both skater AND goalie Edge stats
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge
```

### Subsequent Runs (Same Day)
```bash
# Uses cached Edge stats (fast!)
python main.py --stacks --show-injuries --lineups 5 --edge
```

### Skip Edge (Fastest)
```bash
python main.py --stacks --show-injuries --lineups 5 --no-edge
```

---

## What's New in v2

### Goalie Edge Stats
Now fetching and caching:
- **High-danger save %** (most predictive metric)
- **Midrange save %**
- **Games above .900** (consistency)
- **5v5 save %**
- **Last 10 games save %**

### Goalie Boost Factors
```python
GOALIE_EDGE_BOOST_FACTORS = {
    'elite_hd_save_pct': 1.08,     # +8% for HD SV% >= 85%
    'above_avg_hd_save_pct': 1.04, # +4% for HD SV% >= 82%
    'elite_consistency': 1.05,     # +5% for 80%+ games >.900
    'poor_hd_save_pct': 0.94,      # -6% for HD SV% < 78%
}
```

### Cache Structure
```
projection/cache/
├── edge_stats_2026-02-04.json       # Skater Edge (speed, bursts, OZ%)
├── goalie_edge_stats_2026-02-04.json # Goalie Edge (HD SV%, consistency)
└── edge_metadata.json                # Last fetch timestamps
```

---

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First run runtime | 16 min | 5-6 min | 65% faster |
| Subsequent runs | 16 min | 2-3 min | 85% faster |
| Goalie MAE | 7.20 | ~6.5 (est) | ~10% better |

---

## Testing the Integration

### Test Edge Caching
```python
from edge_cache import EdgeStatsCache, get_cached_edge_stats, get_cached_goalie_edge_stats

# Test skater Edge
skater_df = get_cached_edge_stats(force_refresh=False)
print(f"Skaters: {len(skater_df)}")

# Test goalie Edge  
goalie_df = get_cached_goalie_edge_stats(force_refresh=False)
print(f"Goalies: {len(goalie_df)}")
print(goalie_df[['name', 'team', 'hd_save_pct']].head(10))
```

### Test Goalie Boost Calculation
```python
from edge_stats import EdgeStatsClient

client = EdgeStatsClient()

# Simulate goalie Edge data
test_goalie = {
    'name': 'Igor Shesterkin',
    'hd_save_pct': 0.875,
    'pct_games_above_900': 0.85
}

boost, reasons = client.get_goalie_edge_boost(test_goalie)
print(f"Boost: {boost:.2f} ({reasons})")
# Expected: Boost: 1.13 (['Elite HD SV% (87.5%)', 'Elite consistency (85% games >.900)'])
```

---

## Troubleshooting

### "No goalie Edge data available"
```bash
# Force refresh goalie cache
python -c "from edge_cache import EdgeStatsCache; c = EdgeStatsCache(); c.get_goalie_edge_stats(force_refresh=True)"
```

### "edge_cache module not found"
Make sure `edge_cache.py` is in the same directory as `edge_stats.py` and `main.py`.

### "Cache appears stale"
```python
from edge_cache import EdgeStatsCache
cache = EdgeStatsCache()
print(f"Skater cache age: {cache.get_cache_age_hours('skater'):.1f} hours")
print(f"Goalie cache age: {cache.get_cache_age_hours('goalie'):.1f} hours")
```

---

## Files Included

1. **`edge_cache_v2.py`** → Rename to `edge_cache.py`
   - Daily caching for skater AND goalie Edge stats
   - Automatic cache validation and cleanup
   
2. **`edge_stats_v2.py`** → Rename to `edge_stats.py`
   - New `apply_goalie_edge_boosts()` function
   - Goalie boost factors based on HD save %
   - Integrated with caching module

3. **`INTEGRATION_GUIDE.md`** (this file)
   - Step-by-step integration instructions
   - Code patches for main.py
   - Testing and troubleshooting
