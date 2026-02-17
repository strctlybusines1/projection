#!/usr/bin/env python3
"""
nhl_discord_bot.py — NHL Line Change Monitor
=============================================
Monitors Discord channels for NHL line changes, scratches, goalie
confirmations, and PP unit changes. Parses them into structured data
for the DFS projection pipeline.

Setup:
  1. Create bot at https://discord.com/developers/applications
  2. Enable Message Content Intent
  3. Invite bot to your server
  4. Add NHL news bots or manually post line changes to monitored channels
  5. Put your bot token in .env file

Usage:
  python nhl_discord_bot.py              # Run the bot
  python nhl_discord_bot.py --test       # Test parser on sample messages
"""

import discord
from discord.ext import commands
import os
import re
import json
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Channel names to monitor (bot watches these for line change info)
MONITORED_CHANNELS = [
    'nhl-lines',
    'line-changes',
    'nhl-news',
    'scratches',
    'goalie-confirms',
    'nhl-updates',
]

# Where to save parsed line changes
OUTPUT_DIR = Path(__file__).parent / 'line_changes'
OUTPUT_DIR.mkdir(exist_ok=True)

# Log file for today's changes
def get_today_file():
    return OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_line_changes.json"

def get_today_csv():
    return OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_line_changes.csv"


# ═══════════════════════════════════════════════════════════════
#  NHL LINE CHANGE PARSER
# ═══════════════════════════════════════════════════════════════

# Common NHL team abbreviations
NHL_TEAMS = {
    'ANA','ARI','BOS','BUF','CAR','CBJ','CGY','CHI','COL','DAL',
    'DET','EDM','FLA','LAK','MIN','MTL','NJD','NSH','NYI','NYR',
    'OTT','PHI','PIT','SEA','SJS','STL','TBL','TOR','UTA','VAN',
    'VGK','WPG','WSH',
}

# Position keywords
GOALIE_KEYWORDS = ['starting', 'confirmed', 'in net', 'gets the start', 
                   'will start', 'in goal', 'between the pipes']
SCRATCH_KEYWORDS = ['scratched', 'scratch', 'out tonight', 'will not play',
                    'ruled out', 'won\'t play', 'not in lineup', 'healthy scratch']
LINE_KEYWORDS = ['line 1', 'line 2', 'line 3', 'line 4', 'first line',
                 'second line', 'third line', 'fourth line', 'top line',
                 'top 6', 'top-6', 'top six', 'promoted', 'moved up',
                 'bumped up', 'elevated']
PP_KEYWORDS = ['pp1', 'pp2', 'power play', 'powerplay', 'pp unit',
               'first unit', 'second unit', '1st unit', '2nd unit',
               'man advantage']
INJURY_KEYWORDS = ['injured', 'injury', 'day-to-day', 'dtd', 'ir ',
                   'injured reserve', 'upper body', 'lower body',
                   'game-time decision', 'gtd']


class LineChangeParser:
    """Parse Discord messages into structured line change data."""
    
    def parse(self, message_text, author='', timestamp=None):
        """
        Parse a message and extract NHL line change information.
        
        Returns list of dicts with parsed changes, or empty list if
        no relevant info found.
        """
        text = message_text.strip()
        text_lower = text.lower()
        
        # Skip very short messages or non-hockey content
        if len(text) < 10:
            return []
        
        changes = []
        
        # Check for goalie confirmations
        if any(kw in text_lower for kw in GOALIE_KEYWORDS):
            change = self._parse_goalie(text, text_lower)
            if change:
                change['source'] = author
                change['timestamp'] = timestamp or datetime.now().isoformat()
                change['raw'] = text
                changes.append(change)
        
        # Check for scratches
        if any(kw in text_lower for kw in SCRATCH_KEYWORDS):
            change = self._parse_scratch(text, text_lower)
            if change:
                change['source'] = author
                change['timestamp'] = timestamp or datetime.now().isoformat()
                change['raw'] = text
                changes.append(change)
        
        # Check for line changes
        if any(kw in text_lower for kw in LINE_KEYWORDS):
            change = self._parse_line_change(text, text_lower)
            if change:
                change['source'] = author
                change['timestamp'] = timestamp or datetime.now().isoformat()
                change['raw'] = text
                changes.append(change)
        
        # Check for PP unit changes
        if any(kw in text_lower for kw in PP_KEYWORDS):
            change = self._parse_pp_change(text, text_lower)
            if change:
                change['source'] = author
                change['timestamp'] = timestamp or datetime.now().isoformat()
                change['raw'] = text
                changes.append(change)
        
        # Check for injuries
        if any(kw in text_lower for kw in INJURY_KEYWORDS):
            change = self._parse_injury(text, text_lower)
            if change:
                change['source'] = author
                change['timestamp'] = timestamp or datetime.now().isoformat()
                change['raw'] = text
                changes.append(change)
        
        return changes
    
    def _extract_team(self, text):
        """Extract team abbreviation from text."""
        text_upper = text.upper()
        # Use word boundary matching to prevent substring matches
        # (e.g., ANA inside "PANARIN", CAR inside "SCARY")
        for team in NHL_TEAMS:
            if re.search(r'(?<![A-Z])' + team + r'(?![A-Z])', text_upper):
                return team
        # Check hashtags like #GoAvsGo, #NYR, #LetsGoOilers
        hashtags = re.findall(r'#(\w+)', text)
        for tag in hashtags:
            tag_upper = tag.upper()
            for team in NHL_TEAMS:
                if re.search(r'(?<![A-Z])' + team + r'(?![A-Z])', tag_upper):
                    return team
        return None
    
    def _extract_player_names(self, text):
        """Extract likely player names (capitalized words)."""
        # Pattern handles: Connor McDavid, Nathan MacKinnon, J.T. Miller, 
        # Ryan O'Reilly, De Haan, van Riemsdyk
        names = re.findall(
            r"(?:[A-Z][a-z]*\.?\s+)"                # First name (or initial like J.T.)
            r"(?:(?:Mc|Mac|De|Van|van|O')[A-Z]"      # Prefixed surnames
            r"|[A-Z])"                                # Or normal capital start
            r"[a-zA-Z'-]+",                           # Rest of surname
            text
        )
        # Clean up leading/trailing whitespace
        names = [n.strip() for n in names]
        # Filter out common non-names
        noise = {'The Game', 'Line One', 'Line Two', 'Power Play', 'Daily Faceoff',
                 'Morning Skate', 'Game Time', 'Upper Body', 'Lower Body',
                 'Breaking News', 'Top Line', 'First Line', 'Second Line'}
        return [n for n in names if n not in noise and len(n) > 3]
    
    def _parse_goalie(self, text, text_lower):
        """Parse goalie confirmation."""
        team = self._extract_team(text)
        names = self._extract_player_names(text)
        
        return {
            'type': 'GOALIE_CONFIRM',
            'team': team,
            'player': names[0] if names else None,
            'detail': 'confirmed starter',
        }
    
    def _parse_scratch(self, text, text_lower):
        """Parse scratch notification."""
        team = self._extract_team(text)
        names = self._extract_player_names(text)
        
        return {
            'type': 'SCRATCH',
            'team': team,
            'player': names[0] if names else None,
            'detail': 'scratched',
        }
    
    def _parse_line_change(self, text, text_lower):
        """Parse line promotion/demotion."""
        team = self._extract_team(text)
        names = self._extract_player_names(text)
        
        # Try to extract line number
        line_num = None
        for pattern in [r'line\s*(\d)', r'(\d)(?:st|nd|rd|th)\s*line']:
            match = re.search(pattern, text_lower)
            if match:
                line_num = int(match.group(1))
                break
        
        direction = 'PROMOTED' if any(kw in text_lower for kw in 
                    ['promoted', 'moved up', 'bumped up', 'elevated', 'top line',
                     'first line', 'line 1']) else 'LINE_CHANGE'
        
        return {
            'type': direction,
            'team': team,
            'player': names[0] if names else None,
            'line': line_num,
            'detail': f'moved to line {line_num}' if line_num else 'line change',
        }
    
    def _parse_pp_change(self, text, text_lower):
        """Parse power play unit change."""
        team = self._extract_team(text)
        names = self._extract_player_names(text)
        
        pp_unit = None
        if any(kw in text_lower for kw in ['pp1', 'first unit', '1st unit']):
            pp_unit = 1
        elif any(kw in text_lower for kw in ['pp2', 'second unit', '2nd unit']):
            pp_unit = 2
        
        return {
            'type': 'PP_CHANGE',
            'team': team,
            'player': names[0] if names else None,
            'pp_unit': pp_unit,
            'detail': f'PP{pp_unit}' if pp_unit else 'PP unit change',
        }
    
    def _parse_injury(self, text, text_lower):
        """Parse injury update."""
        team = self._extract_team(text)
        names = self._extract_player_names(text)
        
        status = 'INJURED'
        if 'game-time' in text_lower or 'gtd' in text_lower:
            status = 'GTD'
        elif 'day-to-day' in text_lower or 'dtd' in text_lower:
            status = 'DTD'
        elif 'ir' in text_lower or 'injured reserve' in text_lower:
            status = 'IR'
        
        return {
            'type': status,
            'team': team,
            'player': names[0] if names else None,
            'detail': status,
        }


# ═══════════════════════════════════════════════════════════════
#  STORAGE — Save parsed changes to JSON + CSV
# ═══════════════════════════════════════════════════════════════

class ChangeStore:
    """Persist parsed line changes for the projection pipeline."""
    
    def __init__(self):
        self.changes = []
        self._load_today()
    
    def _load_today(self):
        """Load existing changes from today's file."""
        f = get_today_file()
        if f.exists():
            with open(f) as fh:
                self.changes = json.load(fh)
    
    def add(self, change):
        """Add a new change and save."""
        self.changes.append(change)
        self._save()
    
    def _save(self):
        """Save to both JSON and CSV."""
        # JSON
        with open(get_today_file(), 'w') as fh:
            json.dump(self.changes, fh, indent=2, default=str)
        
        # CSV for easy pipeline consumption
        if self.changes:
            keys = ['timestamp', 'type', 'team', 'player', 'detail', 
                    'line', 'pp_unit', 'source']
            with open(get_today_csv(), 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.changes)
    
    def get_changes(self, change_type=None, team=None):
        """Query today's changes."""
        results = self.changes
        if change_type:
            results = [c for c in results if c.get('type') == change_type]
        if team:
            results = [c for c in results if c.get('team') == team]
        return results
    
    def get_promotions(self):
        """Get all line promotions (most DFS-relevant)."""
        return [c for c in self.changes 
                if c.get('type') in ('PROMOTED', 'PP_CHANGE', 'LINE_CHANGE')]
    
    def get_scratches(self):
        """Get all scratches."""
        return [c for c in self.changes 
                if c.get('type') in ('SCRATCH', 'INJURED', 'IR', 'DTD')]
    
    def get_goalie_confirms(self):
        """Get confirmed goalies."""
        return [c for c in self.changes if c.get('type') == 'GOALIE_CONFIRM']


# ═══════════════════════════════════════════════════════════════
#  DISCORD BOT
# ═══════════════════════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!nhl ', intents=intents)
parser = LineChangeParser()
store = ChangeStore()


@bot.event
async def on_ready():
    print(f'  [NHL Bot] Logged in as {bot.user}')
    print(f'  [NHL Bot] Monitoring channels: {MONITORED_CHANNELS}')
    print(f'  [NHL Bot] Saving to: {OUTPUT_DIR}')


@bot.event
async def on_message(message):
    # Don't respond to ourselves
    if message.author == bot.user:
        return
    
    # Check if message is in a monitored channel
    channel_name = message.channel.name.lower() if hasattr(message.channel, 'name') else ''
    
    # Monitor all channels or specific ones
    should_parse = (
        channel_name in MONITORED_CHANNELS or
        not MONITORED_CHANNELS  # if empty, monitor all
    )
    
    if should_parse:
        changes = parser.parse(
            message.content,
            author=str(message.author),
            timestamp=message.created_at.isoformat()
        )
        
        for change in changes:
            store.add(change)
            
            # Announce the parsed change
            emoji = {
                'GOALIE_CONFIRM': '🥅',
                'SCRATCH': '❌',
                'PROMOTED': '⬆️',
                'LINE_CHANGE': '🔄',
                'PP_CHANGE': '⚡',
                'INJURED': '🏥',
                'DTD': '⚠️',
                'GTD': '⚠️',
                'IR': '🏥',
            }.get(change['type'], '📋')
            
            player = change.get('player', 'Unknown')
            team = change.get('team', '???')
            detail = change.get('detail', '')
            
            await message.channel.send(
                f"{emoji} **Parsed:** {player} ({team}) — {detail}"
            )
    
    # Process commands
    await bot.process_commands(message)


# ── Bot Commands ──

@bot.command(name='status')
async def status(ctx):
    """Show today's tracked changes."""
    n_total = len(store.changes)
    n_promo = len(store.get_promotions())
    n_scratch = len(store.get_scratches())
    n_goalie = len(store.get_goalie_confirms())
    
    msg = (f"📊 **Today's Line Changes**\n"
           f"Total: {n_total}\n"
           f"Promotions/Line Changes: {n_promo}\n"
           f"Scratches/Injuries: {n_scratch}\n"
           f"Goalie Confirms: {n_goalie}")
    await ctx.send(msg)


@bot.command(name='promotions')
async def promotions(ctx):
    """Show today's promotions."""
    promos = store.get_promotions()
    if not promos:
        await ctx.send("No promotions tracked today.")
        return
    
    lines = ["⬆️ **Today's Promotions:**"]
    for p in promos[-10:]:  # last 10
        lines.append(f"  {p.get('player','?')} ({p.get('team','?')}) — {p.get('detail','')}")
    await ctx.send('\n'.join(lines))


@bot.command(name='scratches')
async def scratches(ctx):
    """Show today's scratches."""
    s = store.get_scratches()
    if not s:
        await ctx.send("No scratches tracked today.")
        return
    
    lines = ["❌ **Today's Scratches:**"]
    for p in s[-10:]:
        lines.append(f"  {p.get('player','?')} ({p.get('team','?')}) — {p.get('detail','')}")
    await ctx.send('\n'.join(lines))


@bot.command(name='goalies')
async def goalies(ctx):
    """Show confirmed goalies."""
    g = store.get_goalie_confirms()
    if not g:
        await ctx.send("No goalie confirmations yet.")
        return
    
    lines = ["🥅 **Confirmed Goalies:**"]
    for p in g:
        lines.append(f"  {p.get('player','?')} ({p.get('team','?')})")
    await ctx.send('\n'.join(lines))


@bot.command(name='test')
async def test_parse(ctx, *, text: str):
    """Test the parser on a message. Usage: !nhl test Connor McDavid moved to line 1 PP1 for EDM"""
    changes = parser.parse(text, author='test', timestamp=datetime.now().isoformat())
    if changes:
        for c in changes:
            await ctx.send(f"✅ Parsed: `{json.dumps(c, indent=2, default=str)}`")
    else:
        await ctx.send("❓ No NHL line change info detected in that message.")


# ═══════════════════════════════════════════════════════════════
#  PIPELINE INTEGRATION — Read changes from projection code
# ═══════════════════════════════════════════════════════════════

def get_todays_line_changes():
    """
    Called from main.py or projections.py to get today's line changes.
    
    Usage:
        from nhl_discord_bot import get_todays_line_changes
        changes = get_todays_line_changes()
        for c in changes['promotions']:
            # boost player projection
    """
    s = ChangeStore()
    return {
        'all': s.changes,
        'promotions': s.get_promotions(),
        'scratches': s.get_scratches(),
        'goalies': s.get_goalie_confirms(),
        'timestamp': datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    ap = argparse.ArgumentParser(description='NHL Discord Line Change Bot')
    ap.add_argument('--test', action='store_true', help='Test parser on sample messages')
    args = ap.parse_args()
    
    if args.test:
        print("Testing parser on sample messages:\n")
        p = LineChangeParser()
        
        samples = [
            "Igor Shesterkin confirmed to start tonight for NYR vs PIT",
            "Artemi Panarin has been scratched tonight. Upper body. #NYR",
            "Connor McDavid moving to line 1 with Hyman and Draisaitl. PP1 as well. EDM",
            "Jake Guentzel promoted to top line and PP1 for TBL",
            "Patrice Bergeron is day-to-day with a lower body injury. BOS",
            "Breaking: Alex Ovechkin moved to line 2. WSH shuffling lines at morning skate",
            "Goalies tonight: Vasilevskiy (TBL), Shesterkin (NYR), Oettinger (DAL)",
            "Filip Forsberg PP1 for NSH tonight. Was on PP2 last game.",
        ]
        
        for msg in samples:
            print(f"  MSG: {msg}")
            changes = p.parse(msg)
            for c in changes:
                print(f"  → {c['type']}: {c.get('player','?')} ({c.get('team','?')}) — {c.get('detail','')}")
            if not changes:
                print(f"  → (no change detected)")
            print()
        
    else:
        if not TOKEN:
            print("ERROR: Set DISCORD_BOT_TOKEN in your .env file")
            print("  echo 'DISCORD_BOT_TOKEN=your_token_here' >> .env")
            exit(1)
        
        print("=" * 50)
        print("  NHL Line Change Discord Bot")
        print("=" * 50)
        bot.run(TOKEN)
