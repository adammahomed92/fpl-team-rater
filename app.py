import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Manual team colour mapping (hex codes)
team_colors = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#D20000",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#CC0000",
    "Liverpool": "#C8102E",
    "Luton": "#FF6600",
    "Man City": "#6CABDD",
    "Man Utd": "#DA291C",
    "Newcastle": "#241F20",
    "Nottingham Forest": "#DD0000",
    "Sheffield Utd": "#EE2737",
    "Tottenham": "#132257",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913"
}

def get_bootstrap_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    return requests.get(url).json()

def get_preseason_squad(entry_id):
    # Pre-season data is stored under the "entry" endpoint
    url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/"
    data = requests.get(url).json()
    return data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/team", methods=["POST"])
def team():
    entry_id = request.form.get("entry_id")
    if not entry_id:
        return jsonify({"error": "No Entry ID provided"}), 400

    try:
        bootstrap = get_bootstrap_data()
        players = {p["id"]: p for p in bootstrap["elements"]}
        teams_data = {t["id"]: t for t in bootstrap["teams"]}
        element_types = {e["id"]: e["singular_name"] for e in bootstrap["element_types"]}

        # Try fetching current gameweek picks
        picks_url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/1/picks/"
        picks_resp = requests.get(picks_url)

        if picks_resp.status_code != 200:
            # Pre-season fallback
            return jsonify({
                "message": "Pre-season: showing pre-season squad.",
                "squad": [],
                "rating": None
            })

        picks_data = picks_resp.json()
        if "picks" not in picks_data:
            return jsonify({
                "message": "No squad data found.",
                "squad": [],
                "rating": None
            })

        rows = []
        for pick in picks_data["picks"]:
            player = players[pick["element"]]
            team_name = teams_data[player["team"]]["name"]
            rows.append({
                "name": player["web_name"],
                "position": element_types[player["element_type"]],
                "team": team_name,
                "color": team_colors.get(team_name, "#CCCCCC"),
                "shirt_icon": f"/static/shirts/{team_name.lower().replace(' ', '_')}.png"
            })

        return jsonify({
            "message": "Team loaded successfully",
            "squad": rows,
            "rating": 85  # Placeholder rating
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
