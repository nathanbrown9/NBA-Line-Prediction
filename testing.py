from nba_api.stats.endpoints import playergamelog, teamgamelogs, commonplayerinfo, CommonTeamRoster
from nba_api.stats.static import players, teams
from nba_api.stats.library.parameters import SeasonTypeAllStar
import pandas as pd
import time
import logging
import requests
from bs4 import BeautifulSoup
import sys

# Setup logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Function to get Player ID by Partial Name (Active Players Only)
def get_player_id_by_partial_name(player_name):
    all_players = players.get_active_players()
    matching_players = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
    if not matching_players:
        raise ValueError(f"No active player found with the name '{player_name}'.")
    if len(matching_players) > 12:
        print("Too many results found. Showing the first 12 matches:")
        matching_players = matching_players[:12]
    if len(matching_players) > 1:
        print("\nMultiple active players found:")
        for idx, p in enumerate(matching_players, 1):
            print(f"{idx}. {p['full_name']}")
        choice = input("Enter the number of the player you want: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matching_players):
            selected = matching_players[int(choice) - 1]
            return selected['id'], selected['full_name']
        else:
            raise ValueError("Invalid selection. Please try again.")
    else:
        return matching_players[0]['id'], matching_players[0]['full_name']

# Function to fetch Player Game Logs
def fetch_player_data_by_name(player_name, season="2024-25"):
    """
    Fetch game logs for a player by name and retrieve their team name.
    """
    player_id, full_name = get_player_id_by_partial_name(player_name)
    logging.info(f"\nFetching game logs for: {full_name} (ID: {player_id})")

    # Fetch the player's game logs
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id, 
        season=season, 
        season_type_all_star="Regular Season"
    )
    df = gamelog.get_data_frames()[0]

    # Fetch the player's team name
    player_team_name = get_team_name_by_player_id(player_id)

    return df, full_name, player_id, player_team_name

# Function to map Team Abbreviation to Team ID
def get_team_id_by_abbreviation(abbreviation):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['abbreviation'] == abbreviation), None)
    return team['id'] if team else None

# Function to find Team Name by Team ID
def find_team_name_by_id(team_id):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['id'] == team_id), None)
    return team['full_name'] if team else None

# Function to map partial or full team name to Team ID
def get_team_id_by_name(name):
    nba_teams = teams.get_teams()
    name = name.lower().strip()
    for team in nba_teams:
        if (name in team['full_name'].lower() or
            name in team['nickname'].lower() or
            name in team['city'].lower()):
            return team['id']
    return None

#Get Team By Name or Id
def get_opponent_team_id(df):
    # Extract opponent team abbreviations from MATCHUP column
    opponent_abbreviations = df['MATCHUP'].apply(lambda x: x.split(' ')[-1]).unique()
    opponent_team_ids = []
    for abbrev in opponent_abbreviations:
        team_id = get_team_id_by_abbreviation(abbrev)
        if team_id:
            opponent_team_ids.append(team_id)

    if not opponent_team_ids:
        print("No opponent teams found in the game logs.")
        return None

    print("\nRecent Opponent Teams:")
    unique_opponents = list(set(opponent_team_ids))
    for idx, team_id in enumerate(unique_opponents, 1):
        team_name = find_team_name_by_id(team_id)
        print(f"{idx}. {team_name} ({team_id})")

    choice = input("Choose an opponent team number OR type the team name (e.g., Lakers): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(unique_opponents):
        return unique_opponents[int(choice) - 1]
    
    # Attempt to resolve typed name
    team_id_from_name = get_team_id_by_name(choice)
    if team_id_from_name:
        return team_id_from_name

    print("Invalid selection or team name.")
    return None


# Function to get Opponent Team Name from Team ID
def get_team_name_by_id(team_id):
    team_name = find_team_name_by_id(team_id)
    return team_name if team_name else "Unknown Team"

# Function to fetch recent opponent team stats
def get_opponent_team_stats(team_id, rankings, season="2024-25", num_games=5):
      # Try to fetch playoff game logs first
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id, 
        season_nullable=season, 
        season_type_nullable=SeasonTypeAllStar.playoffs
    )
    df = logs.get_data_frames()[0]
    
    # If no playoff games, fall back to regular season
    if df.empty:
        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id, 
            season_nullable=season, 
            season_type_nullable=SeasonTypeAllStar.regular_season
        )
        df = logs.get_data_frames()[0]
    
    if df.empty:
        return None

    # Filter only numeric columns for calculations
    numeric_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    team_games = df[numeric_stats].head(num_games).apply(pd.to_numeric, errors='coerce')

     # Print game dates, opponent, and stats
    print("\nGames used for calculating averages:")
    for _, row in df.head(num_games).iterrows():
        print(f"Date: {row['GAME_DATE']}, Opponent: {row['MATCHUP']}, "
              f"PTS: {row['PTS']}, REB: {row['REB']}, AST: {row['AST']}, "
              f"STL: {row['STL']}, BLK: {row['BLK']}")

     # Convert stats to numeric for calculations
    team_games = team_games[numeric_stats].apply(pd.to_numeric, errors='coerce')

    # Calculate averages for last 'num_games'
    team_avg = team_games.mean()
    
    # Add rankings
    team_rankings = {stat: rankings[stat].get(team_id, 'N/A') for stat in numeric_stats}
    
    team_avg = team_avg.round(1).to_dict()
    team_avg.update({f"{stat}_RANK": rank for stat, rank in team_rankings.items()})
    
    return team_avg

#----------Getting Rest Info-------------------
def calculate_rest_days(game_dates):
    game_dates_sorted = sorted(pd.to_datetime(game_dates))
    rest_days = [None]  # First game has no prior game
    for i in range(1, len(game_dates_sorted)):
        delta = (game_dates_sorted[i] - game_dates_sorted[i - 1]).days
        rest_days.append(delta)
    return rest_days


#--------------Home vs Away Matchups-------------------------
def home_vs_away(df):
    # Ensure the DataFrame 'df' is defined and contains the 'MATCHUP' column
    if 'MATCHUP' in df.columns:
        # Create a new column 'is_home' where 1 indicates a home game and 0 indicates an away game
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    else:
        print("Error: The DataFrame does not contain the 'MATCHUP' column.")

    return df


#------------Calculating Blowout Risk----------------
def calculate_point_diff(df):
    df['DIFF'] = df['PLUS_MINUS']  # This already is point differential
    blowout_score = df['DIFF'].rolling(window=5).mean().iloc[-1]
    return blowout_score

def estimate_blowout_risk(team_avg_diff, opp_avg_diff):
    return abs(team_avg_diff - opp_avg_diff) > 10  # arbitrary threshold


def add_contextual_features(player_df, team_schedule_df, rankings):
    # Add rest days
    player_df['REST_DAYS'] = calculate_rest_days(team_schedule_df['GAME_DATE'])
    
    # Add home/away
    player_df['IS_HOME'] = home_vs_away['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Add blowout risk
    player_df['BLOWOUT_RISK'] = calculate_point_diff(team_schedule_df, rankings)
    
    # Placeholder for defensive scheme, injury adjustment
    player_df['DEF_SCHEME'] = 0
    #player_df['USAGE_BOOST'] = get_usage_boost(player_name, out_teammates)
    
    return player_df


# Function to calculate league-wide team rankings
def get_league_team_rankings(season="2024-25"):
    logging.info("\nFetching league-wide team statistics for rankings...")
    logs = teamgamelogs.TeamGameLogs(
        season_nullable=season, 
        season_type_nullable="Regular Season"  # Correct parameter for TeamGameLogs
    )
    df = logs.get_data_frames()[0]
    
    # Aggregate stats per team
    aggregated = df.groupby('TEAM_ID').agg({
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean'
    }).reset_index()
    
    # Calculate rankings
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    rankings = {}
    for stat in stats:
        aggregated[f'{stat}_RANK'] = aggregated[stat].rank(ascending=False, method='min')
        rankings[stat] = dict(zip(aggregated['TEAM_ID'], aggregated[f'{stat}_RANK']))
    
    logging.info("League-wide team rankings calculated.")
    return rankings



# Function to calculate weighted averages of player stats
def calculate_averages(df, num_games=5):
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    stats = [stat for stat in stats if stat in df.columns]
    df[stats + ['MIN']] = df[stats + ['MIN']].apply(pd.to_numeric, errors='coerce')
    avg_minutes = df['MIN'].mean()
    
    def weight_game(row):
        if row['MIN'] < 0.7 * avg_minutes:
            return 0.5
        elif row['MIN'] > 0.85 * avg_minutes:
            return 1.5
        return 1.0
    
    df['WEIGHT'] = df.apply(weight_game, axis=1)
    recent_games = df.head(num_games)
    weighted_avg = (recent_games[stats].multiply(recent_games['WEIGHT'], axis=0)).sum() / recent_games['WEIGHT'].sum()
    p = weighted_avg.get('PTS', 0)
    r = weighted_avg.get('REB', 0)
    a = weighted_avg.get('AST', 0)
    p_r = p + r
    p_a = p + a
    r_a = r + a
    p_r_a = p + r + a
    return {
        'PTS': round(p, 1), 'REB': round(r, 1), 'AST': round(a, 1),
        'STL': round(weighted_avg.get('STL', 0), 1), 'BLK': round(weighted_avg.get('BLK', 0), 1),
        'P+R': round(p_r, 1), 'P+A': round(p_a, 1), 'R+A': round(r_a, 1), 'P+R+A': round(p_r_a, 1)
    }

from nba_api.stats.endpoints import commonteamroster

# Function to get player position with standardized codes
'''
def get_player_position(player_id, season="2024-25"):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        data = info.get_data_frames()[0]
        position_full = data['POSITION'][0].strip()
        
        # Mapping full position names to shorthand codes
        position_map = {
            'G': 'G',
            'SG': 'SG',
            'Shooting Guard': 'SG',
            'SF': 'SF',
            'Shooting Forward': 'SF',
            'PF': 'PF',
            'Power Forward': 'PF',
            'C': 'C',
            'Center': 'C',
            '': '',
            None: ''
        }
        
        # Handle cases where 'POSITION' might be in full name or shorthand
        position_shorthand = position_map.get(position_full, '')
        if not position_shorthand:
            # Attempt to derive shorthand from first letter if possible
            if position_full.upper() in ['G', 'SG', 'SF', 'PF', 'C']:
                position_shorthand = position_full.upper()
            else:
                position_shorthand = ''
        
        return position_shorthand
    except Exception as e:
        logging.error(f"Error fetching position for player ID {player_id}: {e}")
        return None'''
# Global position cache (in-memory)
position_cache = {}

def get_player_position(player_id, season="2024-25", use_cache=True, sleep_time=0.1, player_name=None):
    if use_cache and player_name in position_cache:
        return position_cache[player_name]
    elif use_cache and player_id in position_cache:
        return position_cache[player_id]

    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        data = info.get_data_frames()[0]
        position_full = data['POSITION'][0].strip()

        position_map = {
            'G': 'G', 'Guard': 'G', 'Point Guard': 'G', 'Shooting Guard': 'G',
            'F': 'F', 'Forward': 'F', 'Small Forward': 'F', 'Power Forward': 'F',
            'C': 'C', 'Center': 'C',
            'Guard-Forward': 'G', 'Forward-Guard': 'G',
            'Forward-Center': 'F', 'Center-Forward': 'C',
            '': '', None: ''
        }

        position_shorthand = position_map.get(position_full, '')
        if not position_shorthand:
            if position_full.upper() in ['G', 'SG', 'SF', 'PF', 'C']:
                position_shorthand = position_full.upper()
            else:
                position_shorthand = ''

        # Log the player's name and position after it's defined
        name_to_log = player_name if player_name else player_id
        logging.info(f"{name_to_log} is a {position_shorthand}.")

        # Cache result
        if use_cache:
            position_cache[player_id] = position_shorthand

        time.sleep(sleep_time)
        return position_shorthand
    except Exception as e:
        if player_name:
            logging.error(f"Error fetching position for player {player_name}: {e}")
        else:
            logging.error(f"Error fetching position for player ID {player_id}: {e}")
        return None


def get_same_position_players(position, opponent_team_id, player_team_name, season="2024-25", sleep_time=0.3):
    """
    Fetch players of the same position from the opponent team's roster and their stats
    only for games played against the specified player's team.
    """
    team_roster = commonteamroster.CommonTeamRoster(team_id=opponent_team_id, season=season).get_data_frames()[0]
    same_position_players = []

    for idx, row in team_roster.iterrows():
        player_id = row['PLAYER_ID']
        player_name = row['PLAYER']
        try:
            # Get the player's position
            player_position = get_player_position(player_id, season, use_cache=True, sleep_time=sleep_time, player_name=player_name)
            logging.debug(f"Checking {player_name} ({player_position})")

            if player_position == position:
                # Fetch the player's game logs
                regular = playergamelog.PlayerGameLog(
                    player_id=player_id, 
                    season=season, 

                    season_type_all_star="Regular Season"
                ).get_data_frames()[0]

                playoff = playergamelog.PlayerGameLog(
                    player_id=player_id, 
                    season=season, 
                    season_type_all_star="Playoffs"
                ).get_data_frames()[0]

                # Combine regular and playoff logs
                frames = [f for f in [regular, playoff] if not f.empty]
                if not frames:
                    logging.info(f"{player_name} has no valid game logs.")
                    continue

                player_stats = pd.concat(frames, ignore_index=True)

                # Filter the stats to include only games against the player's team
                team_variants = [
                    f"@ {player_team_name.split()[-1]}", f"vs. {player_team_name.split()[-1]}"
                ]
                filtered_stats = player_stats[player_stats['MATCHUP'].apply(
                    lambda x: any(variant in x for variant in team_variants) if isinstance(x, str) else False
                )]

                if filtered_stats.empty:
                    logging.info(f"{player_name} has no games against {player_team_name}.")
                    continue

                # Add the player and their filtered stats to the list
                same_position_players.append({
                    'id': player_id,
                    'full_name': player_name,
                    'stats': filtered_stats
                })

                # Log the player's information and stats
                logging.info(f"Appended player: {player_name} (ID: {player_id})")
                logging.info(f"Stats against {player_team_name}:\n{filtered_stats[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'STL', 'BLK']]}")

        except Exception as e:
            logging.warning(f"Error fetching position or stats for {player_name}: {e}")
            continue

    logging.info(f"Found {len(same_position_players)} {position}s on the opponent team with stats against {player_team_name}.")
    return same_position_players


# Function to calculate matchup deltas
def get_matchup_deltas(opponent_team_id, position, season="2024-25", player_team_name=None):
    """
    Calculate matchup deltas for players of a specific position against an opponent team.
    """
    opponent_team_name = get_team_name_by_id(opponent_team_id)
    logging.info(f"\nCalculating matchup deltas for position: {position} against {opponent_team_name}...")

    if opponent_team_name is None:
        logging.error("Opponent team name could not be determined.")
        return {stat: 0 for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']}

    # Get players of the same position and their stats
    same_position_players = get_same_position_players(position, opponent_team_id, player_team_name, season)
    logging.info(f"Found {len(same_position_players)} players at position {position}.")

    deltas = {stat: [] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']}

    for player in same_position_players:
        player_id = player['id']
        player_name = player['full_name']
        player_stats = player['stats']

        try:
            # Calculate average stats for the matchup games
            matchup_avg = player_stats[['PTS', 'REB', 'AST', 'STL', 'BLK']].mean()
            season_avg = player_stats[['PTS', 'REB', 'AST', 'STL', 'BLK']].mean()
            delta = matchup_avg - season_avg

            for stat in deltas.keys():
                deltas[stat].append(delta.get(stat, 0))

        except Exception as e:
            logging.error(f"Error processing player {player_name}: {e}")
            continue

    # Compute final average deltas
    avg_deltas = {
        stat: round(pd.Series(values).mean(), 2) if values else 0
        for stat, values in deltas.items()
    }

    logging.info(f"Final matchup deltas for position {position} vs. {opponent_team_name}: {avg_deltas}")
    return avg_deltas





# Function to adjust projected stats based on matchup deltas
def calculate_projected_stats(projected_line, matchup_deltas):
    projected = {}
    for stat, value in projected_line.items():
        # Only adjust base stats, not combined stats like P+R
        if stat in matchup_deltas:
            projected[stat] = round(value + matchup_deltas[stat], 1)
        else:
            projected[stat] = value
    return projected

def get_team_name_by_player_id(player_id):
    """
    Fetch the team name of a player using their player_id.
    """
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        data = info.get_data_frames()[0]
        team_name = data['TEAM_NAME'][0]
        return team_name
    except Exception as e:
        logging.error(f"Error fetching team name for player ID {player_id}: {e}")
        return None

# Main Program
if __name__ == "__main__":
    # Fetch league-wide team rankings once to avoid redundant API calls
    try:
        rankings = get_league_team_rankings(season="2024-25")
    except Exception as e:
        logging.error(f"Failed to fetch league team rankings: {e}")
        sys.exit(1)
    
    while True:
        player_name = input("Enter the player's name or last name (or type 'quit' to exit): ").strip()
        if player_name.lower() in ["quit", "exit"]:
            print("Exiting the program.")
            break

        try:
            num_games_input = input("Enter the number of recent games to analyze (default is 5): ").strip()
            num_games = int(num_games_input) if num_games_input.isdigit() else 5

            season = "2024-25"
            df, full_name, player_id, player_team_name = fetch_player_data_by_name(player_name, season)

            if df.empty:
                print(f"No game logs found for {full_name} in the {season} season.")
                continue

            # Get the player's team name
            if not player_team_name:
                print(f"Could not determine the team for {full_name}.")
                continue

            opponent_team = get_opponent_team_id(df)
            matchup_deltas = {}
            position = None

            if opponent_team:
                opponent_stats = get_opponent_team_stats(opponent_team, rankings, season=season, num_games=num_games)
                if opponent_stats is not None:
                    print("\nOpponent Team Recent Averages and Rankings:")
                    for stat, value in opponent_stats.items():
                        print(f"{stat}: {value}")
                else:
                    print("Could not fetch opponent team stats.")
                
                # Get player's position
                position = get_player_position(player_id, season=season)
                if position:
                    matchup_deltas = get_matchup_deltas(opponent_team, position, season=season, player_team_name=player_team_name)
                    print(f"{full_name}'s position: {position}")
                    print("\nMatchup Deltas:")
                    for stat, delta in matchup_deltas.items():
                        print(f"{stat}: {delta}")
                else:
                    print("Could not determine player position.")
            else:
                print("No opponent team selected for strength analysis.")

            # Calculate projected averages
            projected_line = calculate_averages(df, num_games=num_games)
            
            # Adjust projections based on matchup deltas
            if opponent_team and position and matchup_deltas:
                projected_line = calculate_projected_stats(projected_line, matchup_deltas)
            
            print(f"\nProjected line for {full_name} based on last {num_games} games and matchup:")
            for stat, value in projected_line.items():
                print(f"{stat}: {value}")
            
            # Save projections to CSV
            file_name = f"{full_name.replace(' ', '_')}_projected_stats.csv"
            pd.DataFrame([projected_line]).to_csv(file_name, index=False)
            print(f"Projected stats saved to '{file_name}'.")
        
        except ValueError as e:
            print(e)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        print("\n--------------------------------------\n")