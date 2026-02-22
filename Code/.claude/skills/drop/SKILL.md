# File Drop Handler
1. Find the most recent .zip in ~/Downloads
2. Run `unzip -l` to inspect internal structure
3. Extract to /tmp/drop_staging/
4. Copy .py files to the project src directory, .json to config/
5. Check if any previously applied custom edits need re-application (check git diff HEAD~1 for recent manual changes)
6. Run `python -m pytest` if tests exist
7. `git add -A && git commit -m "Integrate file drop $(date +%H:%M)" && git push`
