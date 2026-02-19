import pandas as pd
import re
import os

os.chdir("/Users/brendanhorlbeck/Desktop/NFLGN")

# Read raw data
df = pd.read_csv('PowerSweepW11.csv')
df['Lineup'] = df['Lineup'].astype(str)

def parse_lineup(lineup_str):
    # Initialize
    players = {
        'QB': None,
        'RB': [],
        'WR': [],
        'TE': None,
        'FLEX': None,
        'DST': None
    }

    # Split by position tags: QB, RB, WR, TE, FLEX, DST
    # Use regex to find all positions and what follows
    pattern = r'\b(QB|RB|WR|TE|FLEX|DST)\s+'
    parts = re.split(pattern, lineup_str)
    parts = [p.strip() for p in parts if p.strip()]

    # Group: [pos, value, pos, value, ...]
    i = 0
    current_pos = None
    pos_values = []

    # Reconstruct: every odd index is a position, even is the following text
    tokens = []
    for j in range(len(parts)):
        if parts[j] in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']:
            if j + 1 < len(parts):
                tokens.append((parts[j], parts[j + 1]))
        else:
            # Handle case where position wasn't captured
            if j > 0 and parts[j - 1] not in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']:
                continue

    # If no tokens, fallback to manual scan
    if not tokens:
        for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST']:
            match = re.search(rf'\b{pos}\s+([A-Za-z\-\.\'\s]+?)(?=\s+[A-Z][a-z]|\s+[A-Z]{{2,}}|$)', lineup_str)
            if match:
                name = match.group(1).strip()
                tokens.append((pos, name))

    # Now process each (pos, text) pair
    for pos, text in tokens:
        text = text.strip()

        # Name extraction: allow 2–3 words, handle "St. Brown", "Jr.", etc.
        name_match = re.search(
            r'^([A-Za-z\-\.\']+(?:\s[A-Za-z\.\-\']+)*)', text
        )
        if not name_match:
            continue

        name = name_match.group(1).strip()

        # Special: DST → extract team name only (not player)
        if pos == 'DST':
            team_match = re.search(r'\b([A-Za-z]+)$', name)
            players['DST'] = team_match.group(1) if team_match else name
            continue

        # For RB, WR: append to list
        if pos == 'RB':
            players['RB'].append(name)
        elif pos == 'WR':
            players['WR'].append(name)
        elif pos == 'TE':
            players['TE'] = name
        elif pos == 'FLEX':
            players['FLEX'] = name
        elif pos == 'QB':
            players['QB'] = name

    # Assign slots
    return {
        'QB': players['QB'],
        'RB1': players['RB'][0] if len(players['RB']) > 0 else None,
        'RB2': players['RB'][1] if len(players['RB']) > 1 else None,
        'WR1': players['WR'][0] if len(players['WR']) > 0 else None,
        'WR2': players['WR'][1] if len(players['WR']) > 1 else None,
        'WR3': players['WR'][2] if len(players['WR']) > 2 else None,
        'TE': players['TE'],
        'FLEX': players['FLEX'],
        'DST': players['DST']  # Now always present
    }

# Apply parsing
try:
    parsed_data = df['Lineup'].apply(parse_lineup)
    parsed_df = pd.DataFrame(parsed_data.tolist())

    # Add original Lineup back
    parsed_df['Lineup'] = df['Lineup'].values

    # Reorder columns
    structured_df = parsed_df[['Lineup', 'QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']]

    # Save
    structured_df.to_csv('W11.25_PowerSweepParsed_NFL_Lineupsv1.csv', index=False)
    print("✅ Parsing complete! Saved to 'W2.25SPYParsed_NFL_Lineupsv1.csv'")
except Exception as e:
    print(f"❌ Error during parsing: {e}")
    raise