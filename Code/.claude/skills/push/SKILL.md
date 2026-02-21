# Push to GitHub

1. cd to /Users/brendanhorlbeck/Desktop/Code (the git repo root, not the projection subdirectory)
2. Run `gh auth status` — if token is expired, run `gh auth login` and wait for user to complete auth
3. Run `git status` to review changes
4. Stage relevant files with `git add` (prefer specific files over -A to avoid committing secrets or large binaries)
5. git commit -m "<summarize changes>"
6. git push origin main
