# Run NHL DFS Pipeline

1. cd to /Users/brendanhorlbeck/Desktop/Code (the git repo root)
2. Activate venv: source .venv/bin/activate
3. Check ~/Downloads for new .csv or .zip files related to DFS/DraftKings/NHL data
4. If zip files found, extract them and copy salary CSVs to ./projection/daily_salaries/
5. Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP errors
6. cd to ./projection/ and run: python main.py --stacks --show-injuries --lineups 5 --edge
7. Report the exported lineups from daily_projections/
