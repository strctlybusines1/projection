# Optimized NHL DFS Workflow with Edge Caching

## The Problem

Your current workflow takes 16+ minutes because:
1. **Edge API calls**: ~150+ players × 0.3s delay = 45+ seconds per category × 3 categories = **2-3 minutes**
2. **Every run re-fetches**: Even if you run projections 5x/day, each run makes the same API calls
3. **No caching**: Data that doesn't change is fetched repeatedly

## The Solution: Daily Edge Cache

Edge stats are **cumulative season totals** that update **once daily** (overnight after games).

```
OLD WORKFLOW (16+ min):
┌─────────────────────────────────────────────────────────┐
│ Run 1: Fetch Edge (3 min) → Process (1 min) → Done     │
│ Run 2: Fetch Edge (3 min) → Process (1 min) → Done     │  ← WASTED
│ Run 3: Fetch Edge (3 min) → Process (1 min) → Done     │  ← WASTED
│ Run 4: Fetch Edge (3 min) → Process (1 min) → Done     │  ← WASTED
│ Run 5: Fetch Edge (3 min) → Process (1 min) → Done     │  ← WASTED
└─────────────────────────────────────────────────────────┘
Total: 20 minutes for 5 runs

NEW WORKFLOW (5 min):
┌─────────────────────────────────────────────────────────┐
│ Run 1: Fetch Edge (3 min) → Cache → Process → Done     │
│ Run 2: Load Cache (0.1s)  → Process (1 min) → Done     │  ← FAST
│ Run 3: Load Cache (0.1s)  → Process (1 min) → Done     │  ← FAST
│ Run 4: Load Cache (0.1s)  → Process (1 min) → Done     │  ← FAST
│ Run 5: Load Cache (0.1s)  → Process (1 min) → Done     │  ← FAST
└─────────────────────────────────────────────────────────┘
Total: 7 minutes for 5 runs (65% faster)
```

## New Daily Workflow

### Morning (First Run of Day) - ~5-6 min

```bash
# Step 1: Fetch and cache Edge stats (only needed once per day)
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge

# This will:
# - Fetch fresh Edge stats from NHL API
# - Cache them to cache/edge_stats_YYYY-MM-DD.json
# - Apply boosts and generate projections
```

### Subsequent Runs (Same Day) - ~2-3 min

```bash
# Uses cached Edge stats automatically
python main.py --stacks --show-injuries --lineups 5 --edge

# Or skip Edge entirely for fastest iteration
python main.py --stacks --show-injuries --lineups 5 --no-edge
```

### Pre-Lock Final Run

```bash
# If you want to ensure you have latest Edge (e.g., if unsure about updates)
python main.py --stacks --show-injuries --lineups 5 --edge --refresh-edge
```

## Implementation

### Option 1: Modify edge_stats.py (Recommended)

Add this to the top of your `edge_stats.py`:

```python
from edge_cache import get_cached_edge_stats, EdgeStatsCache

def get_edge_data(force_refresh=False):
    """Get Edge stats with caching."""
    return get_cached_edge_stats(force_refresh=force_refresh)
```

Then in your main Edge function:

```python
def apply_edge_boosts(projections_df, force_refresh=False):
    """Apply Edge stat boosts to projections."""
    
    # Use cached data (fast!) unless force_refresh
    edge_df = get_edge_data(force_refresh=force_refresh)
    
    if edge_df.empty:
        print("⚠️ No Edge stats available")
        return projections_df
    
    # ... rest of your existing boost logic
```

### Option 2: New CLI Flag in main.py

Add to argument parser:

```python
parser.add_argument('--refresh-edge', action='store_true',
                    help='Force refresh Edge stats from API (otherwise use cache)')
```

Then pass to edge function:

```python
if args.edge:
    projections = apply_edge_boosts(projections, force_refresh=args.refresh_edge)
```

## When Does NHL Update Edge Stats?

**Best estimate:** Around 6 AM ET daily, after all games complete.

### How to Verify Edge Stats Are Updated

```python
from edge_cache import EdgeStatsCache

cache = EdgeStatsCache()

# Check cache age
age = cache.get_cache_age_hours()
print(f"Cache age: {age:.1f} hours")

# If cache is >24 hours old, definitely refresh
if age and age > 24:
    print("Cache is stale, refreshing...")
    edge_df = cache.get_edge_stats(force_refresh=True)
```

### Heuristic for "Should I Refresh?"

```
┌─────────────────────────────────────────────────────────┐
│ Time of Day          │ Action                           │
├─────────────────────────────────────────────────────────┤
│ Before 6 AM ET       │ Use yesterday's cache (still ok) │
│ 6 AM - 10 AM ET      │ Refresh (new day's data ready)   │
│ 10 AM - 6 PM ET      │ Use today's cache                │
│ After 6 PM ET        │ Use today's cache (games live)   │
└─────────────────────────────────────────────────────────┘
```

## Files to Add

1. **`edge_cache.py`** - The caching module (provided)
2. **`cache/`** - Directory for cached files (auto-created)

## Cache File Structure

```
projection/
├── cache/
│   ├── edge_stats_2026-02-04.json   # Today's Edge data
│   ├── edge_stats_2026-02-03.json   # Yesterday's (auto-cleaned after 7 days)
│   └── edge_metadata.json           # Last fetch timestamps
├── edge_stats.py                     # Your existing file (modify to use cache)
├── edge_cache.py                     # New caching module
└── main.py                           # Add --refresh-edge flag
```

## Performance Comparison

| Scenario | Old Time | New Time | Savings |
|----------|----------|----------|---------|
| First run of day | 16 min | 5-6 min | 62% |
| 2nd-5th runs | 16 min each | 2-3 min each | 85% |
| Full day (5 runs) | 80 min | 17 min | 79% |

## Troubleshooting

### "Edge stats not loading"

```bash
# Check if cache exists
ls -la cache/edge_stats_*.json

# Force refresh
python main.py --edge --refresh-edge
```

### "Edge stats seem outdated"

```python
from edge_cache import EdgeStatsCache
cache = EdgeStatsCache()
print(f"Last fetch: {cache._load_metadata().get('last_fetch_timestamp')}")
```

### "Want to clear all caches"

```bash
rm -rf cache/edge_stats_*.json
```

## Summary

| Flag | Behavior |
|------|----------|
| `--edge` | Use Edge stats (cached if available) |
| `--edge --refresh-edge` | Fetch fresh Edge stats, then cache |
| `--no-edge` | Skip Edge entirely (fastest) |
| No flag | No Edge stats |

**Recommended daily workflow:**
1. Morning: `--edge --refresh-edge` (once)
2. Rest of day: `--edge` (uses cache)
3. Pre-lock: `--edge` or `--edge --refresh-edge` if paranoid
