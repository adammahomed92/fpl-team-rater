import requests

def get_team_data(entry_id, gw, cookies=None):
    if gw > 0:  # try gameweek picks
        picks_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
        r = requests.get(picks_url)
        if r.status_code == 200:
            return r.json()
    
    # fallback to preseason squad
    if cookies:
        my_team_url = f"https://fantasy.premierleague.com/api/my-team/{entry_id}/"
        r = requests.get(my_team_url, headers={"Cookie": cookies})
        if r.status_code == 200:
            return r.json()
    
    return None

# Example usage:
entry_id = 2792859
gameweek = 1  # GW1
cookies = "pl_profile=XXXX; pl_session=YYYY"  # copied from browser
data = get_team_data(entry_id, gameweek, cookies)

if data:
    print(data)
else:
    print("No team data found.")
