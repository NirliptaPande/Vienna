import requests
url = 'https://raw.githubusercontent.com/intelligent-environments-lab/CityLearn/citylearn_2022/citylearn/data/citylearn_challenge_2022_phase_1/Building_1.csv'
res = requests.get(url, allow_redirects=True)
with open('b1.csv','wb') as file:
    file.write(res.content)
