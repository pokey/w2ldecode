from dataclasses import dataclass
from typing import Optional, Any
import math
import random
import numpy as np
import kenlm
import os

CONFORMER_PATH = os.path.expanduser("~/.talon/w2l/en_US-conformer")
# specific to double precision
MINUS_LOG_THRESHOLD = -39.14

# Adds log-likelihoods. Returns log(exp(loga) + exp(logb)), ie: given loga =
# log(a), logb = log(b), returns log(a + b).
#
# Think of a, b, and a + b as probabilities. Floating point has limited
# precision, so to manipulate probabilities it is better to represent them by
# their logarithm, which ranges from log(0) = -inf to log(1) = 0.
def log_add(loga, logb):
    if loga < logb: loga, logb = logb, loga
    minusdif = logb - loga
    # I think this is an optimisation to avoid doing work if it wouldn't affect
    # the result due to precision limitation, but I'm not sure.
    if minusdif < MINUS_LOG_THRESHOLD: return loga
    return loga + np.log1p(math.exp(minusdif))

@dataclass
class TryNode:
    score: float
    words: list[tuple[str, float]]
    children: dict[str, 'TryNode']

    def smear_score(self):
        # It is strange to mix log-add and max this way. It's what the original
        # does in SmearMode::MAX, but this may be a mistake. However, since we
        # effectively only use log_add if there's more than one word for a given
        # trie node, it's basically irrelevant for conformer.
        self.score = -math.inf
        for _, score in self.words:
            self.score = log_add(self.score, score)
        for child in self.children.values():
            child.smear_score()
            self.score = max(self.score, child.score)

class Trie:
    root: TryNode

    def __init__(self, *, model_path, lexicon_path):
        self.root = TryNode(score=0, words=[], children={})
        model = kenlm.Model(model_path)
        words = []
        print("loading trie")
        with open(lexicon_path, 'r') as f:
            for i, line in enumerate(f):
                word, *tokens = line.split()
                if (i+1) % 5000 == 0: print(f" {word}...", end='', flush=True)
                self.insert(word, tokens, model.score(word))
        print("\ncalculating scores...")
        self.root.smear_score()
        # just to get an idea of what some trie nodes look like
        for word in ["lexicons", "airport"]:
            node = self.root
            for char in word:
                node = node.children[char]
            print(node)

    def insert(self, word, tokens, score):
        node = self.root
        for t in tokens:
            if t not in node.children:
                node.children[t] = TryNode(score=0, words=[], children={})
            node = node.children[t]
        node.words.append((word, score))

@dataclass
class Tokens:
    token: str
    parent: Optional['Token']

@dataclass
class Beam:
    tokens: Tokens
    state: Any
    emission_score: float

    def score(self):
        # probably need a balancing factor here
        return self.emission_score + self.state.score()

    def text(self):
        node = self.tokens
        out: list[str] = []
        while node is not None:
            out.append(node.token)
            node = node.parent
        return ''.join(reversed(out))

    def __str__(self):
        return f"Beam {{{self.score()}, {self.text()}}}"

class Decoder:
    tokens:    str
    criterion: str
    trie:      Trie

    beamsize: int
    beamthreshold: float

    def __init__(self, tokens: str, *, criterion: str, trie: Trie, beamsize: int=1, beamthreshold: float=math.inf):
        if criterion == 'ctc' and not '#' in tokens:
            tokens += '#'
        self.tokens    = tokens
        self.criterion = criterion
        self.trie      = trie
        self.beamsize  = beamsize
        self.beamthreshold = beamthreshold

    def fake_encode(self, s: str) -> np.array:
        # generates a fake emissions matrix
        if self.criterion == 'ctc':
            tmp: list[str] = []
            # add CTC blanks where necessary
            for i, c in enumerate(s):
                tmp.append(c)
                if i < len(s)-1 and c == s[i+1]:
                    tmp.append('#')
            s = ''.join(tmp)
        s = s.replace(' ', '|')

        out = np.zeros((0, len(self.tokens)))
        for c in s:
            width = random.randint(3, 5) if c != '#' else 1
            for i in range(width):
                x = np.random.rand(1, len(self.tokens)) * 9.99
                x[0,self.tokens.index(c)] = 10
                out = np.concatenate([out, x])
        return out

    def greedy_decode(self, x: np.array, *, short=False) -> str:
        if self.criterion == 'ctc':
            idx = [np.argmax(step) for step in x]
            out = ''.join([self.tokens[i] for i in idx])
            if short:
                out = ''.join([c for i, c in enumerate(out)
                               if i == 0 or c != out[i-1]])
                out = out.replace('#', '').replace('|', ' ')
            return out
        raise NotImplementedError

    def decode(self, x: np.array) -> list[Beam]:
        # performs beam search
        root = Beam(tokens=Tokens('|', None), emission_score=0, try_node=self.trie.root)
        beams: list[Beam] = [root]
        for t, step in enumerate(x):
            new_beams: list[Beam] = []
            for parent in beams:
                prev_token = parent.tokens.token
                # possible next tokens
                suggestions = {prev_token} | parent.try_node.children.keys()
                if self.criterion == 'ctc': suggestions.add('#')
                for n, token in enumerate(self.tokens):
                    if token not in suggestions: continue
                    score = step[n]
                    new_beams.append(Beam(
                        tokens = Tokens(token, parent.tokens),
                        emission_score = score + parent.emission_score,
                        # if token is nonblank & distinct from previous token,
                        # we transition the try node.
                        try_node = parent.try_node
                                   if token == prev_token or token == '#'
                                   else parent.try_node.children[token]
                    ))
            if not new_beams:
                print("STUCK:")
                for b in beams: print(f"   {b}")
                return []
            beams = new_beams
            # prune beamsize
            beams.sort(key=lambda x: x.score(), reverse=True)
            beams = beams[:self.beamsize]
            # prune beamthreshold
            best_score = beams[0].score()
            beams = [b for b in beams if b.score() >= best_score - self.beamthreshold]
            print(f" step {t} best: {beams[0]}")
        return beams

    def synthetic_test(self, s: str) -> None:
        x = self.fake_encode(s)
        print('greedy()     ',  self.greedy_decode(x))
        print('greedy(short)',  self.greedy_decode(x, short=True))
        beams = self.decode(x)
        if beams: print('decode()     ',  beams[0].text())
        else: print('decode() FAILED!')
        for candidate in beams:
            print(f"{candidate.score():.3f} {candidate.text()}")

def main():
    tokens = "|'abcdefghijklmnopqrstuvwxyz"
    trie = Trie(model_path=os.path.join(CONFORMER_PATH, "lm-ngram.bin"),
                lexicon_path=os.path.join(CONFORMER_PATH, "lexicon.txt"))
    decoder = Decoder(tokens, criterion='ctc', beamsize=20, trie=trie)
    decoder.synthetic_test('a')
    decoder.synthetic_test('hello')
    # these don't decode properly yet, only handle single words so far
    decoder.synthetic_test('hello world ')
    decoder.synthetic_test('this is a decoder test ')

if __name__ == '__main__': main()
