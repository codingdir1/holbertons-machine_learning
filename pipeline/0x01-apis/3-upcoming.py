#!/usr/bin/env python3

import requests

def parse_date(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%fZ')

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    if response.status_code == 200:
        json = response.json()

        dates = [x['date_utc'] for x in json]
        index = dates.index(min(dates))
        next_launch = json[index]

        name = next_launch["name"]
        date = next_launch["date_local"]
        rocket = next_launch["rocket"]
        launchpad = next_launch["launchpad"]

        url_rocket = "https://api.spacexdata.com/v4/rockets/" + rocket
        rocket_req = requests.get(url_rocket)
        if rocket_req.status_code == 200:
            json_rocket = rocket_req.json()
            rocket_name = json_rocket["name"]
        
        url_launchpad = "https://api.spacexdata.com/v4/launchpads/" + launchpad
        launchpad_req = requests.get(url_launchpad)
        if launchpad_req.status_code == 200:
            json_launchpad = launchpad_req.json()
            launchpad_name = json_launchpad["name"]
            launchpad_loc = json_launchpad["locality"]

        print("{0} ({1}) {2} - {3} ({4})".format(name, date, rocket_name, launchpad_name, launchpad_loc))
