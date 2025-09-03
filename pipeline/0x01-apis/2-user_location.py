#!/usr/bin/env python3

import requests
import sys
import time

if __name__ == "__main__":
    url = sys.argv[1]

    response = requests.get(url)

    if (response.status_code == 404):
        print("Not found")
    elif (response.status_code == 403):
        ratelimit = int(response.header["X-Ratelimit-Reset"])
        now = int(time.time())
        print("Reset in {0} min".format(ratelimit - now))
    else:
        print(response.json()["location"])
