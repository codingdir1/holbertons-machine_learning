#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ["Farrah", "Fred", "Felicia"]
fruit_labels = ["Apples", "Bananas", "Oranges", "Peaches"]
colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

x = np.arange(len(people))

bottom = np.zeros(len(people))
for i in range(fruit.shape[0]):
    plt.bar(x, fruit[i], bottom=bottom, label=fruit_labels[i], color=colors[i], width = 0.5)
    bottom += fruit[i]

plt.xticks(x, people)
plt.ylim(0, 80)
plt.yticks(list(range(0, 81, 10)))
plt.ylabel("Quantity of Fruit")
plt.title("Number of fruits per person")
plt.legend()
plt.show()
