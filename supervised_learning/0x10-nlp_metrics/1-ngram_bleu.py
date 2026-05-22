#!/usr/bin/env python3

import numpy as np

def ngram_bleu(references, sentence, n):
    bleuScore = 0
    refLen = len(references)
    sentLen = len(sentence)

    nGramRefs = []
    for reference in references:
        nGramRef = [" ". join(reference[i : i + n]) for i in range(0, len(reference) - n + 1, 1)]
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

    nGramSent = [" ".join(sentence[i : i + n]) for i in range(0, sentLen - n + 1, 1)]
    nGramSentDict = {}
    for word in nGramSent:
        nGramSentDict[word] = nGramSentDict.get(word, 0) + 1

    score = 0
    for word, count in nGramSentDict.items():
        if word in nGramRefDict.keys():
            score += min(count, max(nGramRefDict[word]))

    if len(nGramSent) == 0 or score == 0:
        return 0.0

    bleuScore = score / len(nGramSent)
   
    refSentLen = [len(references[i]) for i in range(0, refLen)]
    closestSentLen = min(refSentLen, key = lambda x : (abs(sentLen - x)))
    if sentLen > closestSentLen:
        brevity = 1
    else:
        brevity = np.exp(1 - (closestSentLen / sentLen))
    return bleuScore * brevity
