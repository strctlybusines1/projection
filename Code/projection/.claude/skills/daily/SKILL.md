# Daily NHL DFS Pipeline
1. cd to project root
2. Run: python scrapers.py
3. Run: python lines.py
4. Run: python projections.py
5. Run: python main.py
6. Save all outputs to disk and verify files exist
7. Start dashboard on port 5050: python app.py &
8. Verify dashboard is responding: curl -s http://localhost:5050
9. git add . && git commit -m "Daily pipeline run $(date +%Y-%m-%d)" && git push
