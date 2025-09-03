#!/usr/bin/env python3

import requests

def sentientPlanets():
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    params = {"formal" : "json"}
    home_url_list = []
    home_list = []

    while url != None:
        response = requests.get(url, params = params)
        if response.status_code == 200:
            data = response.json()
            for species in data["results"]:
                if species["designation"] == "sentient":
                    if species["homeworld"] != None:
                        home_url_list.append(species["homeworld"])
        url = data["next"]
    for home in home_url_list:
        response = requests.get(home, params = params)
        if (response.status_code == 200):
            data = response.json()
            home_list.append(data["name"])
    return home_list
