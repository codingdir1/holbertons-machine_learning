#!/usr/bin/env python3

import numpy as np

def cumulative_bleu(references, sentence, n):
    bleuScore = 0
    refLen = len(references)
    sentLen = len(sentence)
    logPrecisions = []
    for N in range(1, n + 1):
        nGramRefs = []
        for reference in references:
            nGramRef = [" ". join(reference[i : i + N]) for i in range(0, len(reference) - N + 1, 1)]
            nGramRefs.append(nGramRef)

        nGramRefDict = {}
        nGramRefLen = len(nGramRefs)
        for i in range(0, nGramRefLen):
            for word in nGramRefs[i]:
                if word not in nGramRefDict.keys():
                    nGramRefDict[word] = [0] * nGramRefLen
                    nGramRefDict[word][i] = 1
                else:
                    nGramRefDict[word][i] += 1

        nGramSent = [" ".join(sentence[i : i + N]) for i in range(0, sentLen - N + 1, 1)]
        nGramSentDict = {}
        for word in nGramSent:
            nGramSentDict[word] = nGramSentDict.get(word, 0) + 1

        score = 0
        for word, count in nGramSentDict.items():
            if word in nGramRefDict.keys():
                score += min(count, max(nGramRefDict[word]))

        if len(nGramSent) == 0 or score == 0:
            return 0.0
        logPrecisions.append(np.log(score / len(nGramSent)))
    bleuScore = np.exp(np.mean(logPrecisions))
   
    refSentLen = [len(references[i]) for i in range(0, refLen)]
    closestSentLen = min(refSentLen, key = lambda x : (abs(sentLen - x)))
    if sentLen > closestSentLen:
        brevity = 1
    else:
        brevity = np.exp(1 - (closestSentLen / sentLen))
    return bleuScore * brevity
