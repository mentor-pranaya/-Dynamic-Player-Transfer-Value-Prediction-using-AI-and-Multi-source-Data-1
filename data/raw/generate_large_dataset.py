import pandas as pd
import numpy as np
import random

first_names = ["Lionel","Cristiano","Kylian","Erling","Kevin","Robert","Neymar","Luka","Harry","Karim","Vinicius","Jude","Sadio","Mohamed","Ruben","Pedri","Gavi","Joao","Marcus","Bruno","Federico","Paul","Marco","Bernardo","Thomas","Ilkay","Riyad","Phil","Darwin","Luis","Diogo","Sergio","David","Alisson","Ederson","Jan","Manuel","Marc","Andre","Angel","Rodrygo","Eduardo","Benjamin","Thiago","Achraf","Mason","Declan","Ferran","Kai","Reece","Josko"]
last_names = ["Messi","Ronaldo","Mbappe","Haaland","De Bruyne","Lewandowski","Junior","Modric","Kane","Benzema","Silva","Salah","Mane","Dias","Felix","Rashford","Fernandes","Valverde","Pogba","Reus","Alcantara","Mahrez","Foden","Nunez","Suarez","Jota","Busquets","Becker","Mendy","Courtois","Ter Stegen","Camavinga","Di Maria","Pavard","Hernandez","Mount","Rice","Gundogan","Olmo","Musiala","Nkunku","Havertz"]
positions = ["ST","LW","RW","CM","CAM","CB","LB","RB","GK","CDM","LM","RM","CF"]
leagues = ["Premier League","La Liga","Serie A","Bundesliga","Ligue 1","Eredivisie","Primeira Liga","Saudi League"]
clubs = ["Barcelona","Real Madrid","Liverpool","Manchester City","Manchester United","PSG","Bayern Munich","Dortmund","Juventus","Inter Milan","AC Milan","Chelsea","Arsenal","Atletico Madrid","Sevilla","Ajax","Porto","Benfica","Al Nassr","Al Hilal"]
nationalities = ["Spain","Portugal","Brazil","Argentina","France","Germany","England","Italy","Netherlands","Belgium","Uruguay","Croatia","Norway","Morocco","Japan","Korea","USA","Chile","Colombia"]

players = []

for i in range(10000):
    name = random.choice(first_names) + " " + random.choice(last_names)
    age = random.randint(17, 40)
    nationality = random.choice(nationalities)
    league = random.choice(leagues)
    club = random.choice(clubs)
    position = random.choice(positions)

    base = np.random.normal(70, 10)
    if club in ["Barcelona","Real Madrid","PSG","Manchester City","Bayern Munich"]:
        base += np.random.normal(5, 3)

    overall = int(np.clip(base, 50, 95))
    potential = int(np.clip(overall + np.random.normal(3, 5), overall, 99))

    pace = int(np.clip(np.random.normal(overall, 10), 35, 99))
    shooting = int(np.clip(np.random.normal(overall, 10), 35, 99))
    passing = int(np.clip(np.random.normal(overall, 10), 35, 99))
    dribbling = int(np.clip(np.random.normal(overall, 10), 35, 99))
    physical = int(np.clip(np.random.normal(overall, 10), 35, 99))

    matches = random.randint(10, 50)
    goals = max(0, int(np.random.normal(matches / 3, 2)))
    assists = max(0, int(np.random.normal(matches / 4, 2)))
    minutes = matches * random.randint(50, 90)
    injury_count = random.randint(0, 5)

    value = (overall * 600000) + (potential * 400000) - (age * 20000) + (goals * 50000)
    value = int(max(100000, value))

    players.append([
        name, age, nationality, club, league, position,
        overall, potential, pace, shooting, passing,
        dribbling, physical, matches, goals, assists,
        minutes, injury_count, value
    ])

df = pd.DataFrame(players, columns=[
    "player_name","age","nationality","club","league","position",
    "overall","potential","pace","shooting","passing",
    "dribbling","physical","matches","goals","assists",
    "minutes","injury_count","market_value"
])

df.to_csv("players_full_10000.csv", index=False)
print("Generated: players_full_10000.csv")
