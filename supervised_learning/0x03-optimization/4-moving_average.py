#!/usr/bin/env python3

def moving_average(data, beta):
    data_prev = 0
    moving_avg = []
    for i in range(len(data)):
        data_prev = beta * data_prev + (1 - beta) * data[i]
        moving_avg.append(data_prev / (1 - beta ** (i + 1)))
    return moving_avg

