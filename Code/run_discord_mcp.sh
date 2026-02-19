#!/usr/bin/env bash
# Wrapper for discord-mcp-server: loads projection/.env then runs the server.
# Keeps DISCORD_TOKEN and DISCORD_GUILD_ID out of mcp.json.
set -e
export PATH="/opt/homebrew/bin:$PATH"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
set -a
[ -f projection/.env ] && source projection/.env
set +a
exec npx --yes @ncodelife/discord-mcp-server
