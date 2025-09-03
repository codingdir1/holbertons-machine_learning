#!/usr/bin/env python3

from datetime import datetime
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url)

    if  (response.status_code == 200):
        json = response.json()

    rocketDict = {}
    for launch in json:
        rocket = launch['rocket']
        launch_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        launch_response = requests.get(launch_url)
        if launch_response.status_code == 200:
            launch_json = launch_response.json()

        rocket_name = launch_json["name"]
        if rocketDict.get(rocket_name) is None:
            rocketDict[rocket_name] = 1
        else:
            rocketDict[rocket_name] += 1
        rocketList = sorted(rocketDict.items(), key=lambda kv: kv[0])
        rocketList = sorted(rocketList, key=lambda kv: kv[1], reverse=True)
        for rocket in rocketList:
            print("{}: {}".format(rocket[0], rocket[1]))
