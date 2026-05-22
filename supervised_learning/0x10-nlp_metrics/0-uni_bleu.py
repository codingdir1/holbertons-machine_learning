#!/usr/bin/env python3

import numpy as np

def uni_bleu(references, sentence):
    bleuScore = 0
    refLen = len(references)
    sentLen = len(sentence)

    refDict = {}
    for i in range(0, refLen):
        for word in references[i]:
            if word not in refDict.keys():
                refDict[word] = [0] * refLen
                refDict[word][i] = 1
            else:
                refDict[word][i] += 1

    sentDict = {}
    for word in sentence:
        sentDict[word] = sentDict.get(word, 0) + 1

    score = 0
    for word, count in sentDict.items():
        if word in refDict.keys():
            score += min(count, max(refDict[word]))

    if len(sentence) == 0 or score == 0:
        return 0.0

    bleuScore = score / len(sentence)
   
    refSentLen = [len(references[i]) for i in range(0, refLen)]
    closestSentLen = min(refSentLen, key = lambda x : (abs(sentLen - x)))
    if sentLen > closestSentLen:
        brevity = 1
    else:
        brevity = np.exp(1 - (closestSentLen / sentLen))
    return bleuScore * brevity
