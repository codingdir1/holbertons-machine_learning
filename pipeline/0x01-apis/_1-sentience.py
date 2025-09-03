#!/usr/bin/env python3
"""
1. Where I am?
"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    species_list = []
    while url:
        species_results = requests.get(url).json()
        species_list += species_results.get('results')
        url = species_results.get('next')
    planets = []
    for species in species_list:
        if species.get('designation') == 'sentient' or species.get('classification') == 'sentient':
          url = species.get('homeworld')
          if url:
             planet = requests.get(url).json()
             planets.append(planet.get('name'))
             return planets
print(sentientPlanets())
