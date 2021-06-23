from dataclasses import dataclass
from typing import Optional, Any, Iterator
import math
import random
import numpy as np
import kenlm
import os

CONFORMER_PATH = os.path.expanduser("~/.talon/w2l/en_US-conformer")

# TODOs
# 1. use ints for tokens instead of strs
#
# 2. fix the scoring. refer to the c++ code.
#    should make a comment explaining the scoring once I grok it.
# 2a. mixing scores from model/emissions needs a weighting factor
# 2b. various places need hyperparameters mixed in (finishing a word, silence)
#
# 3. factor out the LM stuff, like in aegis' example, instead of inlining kenlm code everywhere


# prevLmState is DFALM::State from w2l_decode_backend.cpp -- AM I SURE?
# because DFALM::State::maxWordScore always returns 0, which seems very wrong.
# what is getPrevSil()?

# CombinedDecoder = SimpleDecoder<DFALM::LM, DFALM::State>
# SimpleDecoder<lm, lmstate> -> BeamSearch<lm, lmstate>

# prevLmState.maxWordScore
# prevLmState = prevHyp.lmState
# prevHyp comes from hypIt comes from range() comes from hyp
# hyp: vector<DecoderState>
# BeamSearch<lm, lmstate>::DecoderState = SimpleDecoderState<lmstate>
# SimpleDecoderState<lmstate>.lmState: lmstate
# ie. it's a DFALM::State ?!?


# ---------- HOW SCORING WORKS ----------
# We generate new beams from these sources:
#
# 1. For each child of prevLmState (which is a DFA guide):
#
#    score += emission[tok] + (silScore if tok is silence or blank)
#
#    (1a) if found word(s)
#    score += wordScore + lmWeight * lmDelta
#      where lmDelta = ??? i think this is determined by 
#
#    (1b) if has children
#    score += lmWeight * lmDelta
#
# 2. Try repeating previous token (even when not in ctc?)
#    does something weird where it pretends tok is silence sometimes?
#
#    score += emission[tok] + (silScore if tok is silence or blank)
#
# 3. If CTC, try blank token.
#
#    score += emission[tok] + silScore
#
#
# UNIFIED SCORING RULE:
#
#    score += emission[tok], always
#    score += silScore, if tok is silence or blank
#    score += wordScore, if found a word
#    score += lmWeight * lmDelta, which will be 0 if we didn't transition the graph node
#
# I think lmDelta is determined by the DFA/LM/Trie stack, not by just the LM, despite name.

# Guide is an interfaces. TODO: how do those work in mypy?
class Guide:
    def children(self, beam: 'Beam', decoder: 'Decoder') -> Iterator['Beam']:
        raise NotImplementedError

GuideStack = Optional[tuple[Guide, 'GuideStack']]

@dataclass
class Beam:
    # previous/parent beam state
    parent: Optional['Beam']
    token: str
    score: float
    # LM state, preserved across multiple invocations of LM
    lm_state: kenlm.State
    # state used to guide the search
    guide: Guide
    guides: GuideStack

    def child(self, token, score_delta = 0) -> 'Beam':
        # the point of putting score_delta here is that scores coming from
        # emissions and scores coming from model/graph/dfa should be weighted
        # differently eventually and this was the only place I could find to
        # centralize contributions from the model/graph/dfa. kinda ugly :/
        return Beam(parent = self, token = token, score = self.score + score_delta,
                    lm_state = self.lm_state, guide=self.guide, guides=self.guides)

    def text(self):
        node = self
        out: list[str] = []
        while node is not None:
            out.append(node.token)
            node = node.parent
        return ''.join(reversed(out))

    # for debugging
    def __str__(self):
        guides = [self.guide]
        stack = self.guides
        while stack:
            guide, stack = stack
            guides.append(guide)
        return f"Beam {self.text()} {self.score:.3f} {' '.join([str(x) for x in guides])}"

# implements Guide
@dataclass
class TryNode:
    score: float
    words: list[tuple[str, float]]
    edges: dict[str, 'TryNode']

    def __str__(self):
        return f"(TryNode {self.score:.3f} {self.words} {''.join(self.edges.keys())})"

    def children(self, beam: Beam, decoder: 'Decoder') -> Iterator[Beam]:
        for token, node in self.edges.items():
            if node.edges:      # avoid dead-ending
                b = beam.child(token, node.score - self.score)
                b.guide = node
                yield b
            for word, _ in node.words:
                assert token == '|'
                new_lm_state = kenlm.State()
                score = decoder.model.BaseScore(beam.lm_state, word, new_lm_state)
                # TODO: probably want to add opt_.word_score here/somewhere?
                # should this be score or (score - node.score)?
                b = beam.child(token, score - node.score)
                # print(f"found word: {word}, delta {score - self.score:.3f}, score {b.score}")
                b.guide, b.guides = beam.guides
                b.lm_state = new_lm_state
                yield b

@dataclass
class LMGuide:
    def children(self, beam: Beam, decoder: 'Decoder') -> Iterator[Beam]:
        # TODO: does this need an infusion of opt_.silScore?
        yield beam.child('|')
        # a hack to descend into the trie
        b = Beam(parent=beam.parent,
                 token=beam.token,
                 score=beam.score,
                 lm_state=beam.lm_state,
                 guide = decoder.trie,
                 guides = (beam.guide, beam.guides))
        yield from decoder.trie.children(b, decoder)

class Decoder:
    tokens:    str
    model:     kenlm.Model
    trie:      TryNode

    beamsize: int
    beamthreshold: float

    def __init__(self, tokens: str, *,
                 model: kenlm.Model, trie: TryNode,
                 beamsize: int=1, beamthreshold: float=math.inf):
        assert '#' not in tokens
        tokens += '#'
        self.tokens    = tokens
        self.model     = model
        self.trie      = trie
        self.beamsize  = beamsize
        self.beamthreshold = beamthreshold

    def decode(self, x: np.array) -> list[Beam]:
        # performs beam search
        lm_state = kenlm.State()
        self.model.BeginSentenceWrite(lm_state)
        guide = LMGuide()
        root = Beam(parent=None, token='|', score=0, lm_state=lm_state, guide=guide, guides=None)
        beams: list[Beam] = [root]

        for t, step in enumerate(x):
            new_beams: list[Beam] = []
            for parent in beams:
                prev_token = parent.token
                for child in parent.guide.children(parent, self):
                    if child.token == prev_token: continue
                    assert child.token != '#'
                    child.score += step[self.tokens.index(child.token)]
                    new_beams.append(child)
                for token in ['#', prev_token]:
                    # TODO: need a silence score here for '#'?
                    beam = parent.child(token)
                    beam.score += step[self.tokens.index(token)]
                    new_beams.append(beam)
            if not new_beams:
                print("STUCK:")
                for b in beams: print(f"   {b}")
                return []
            beams = new_beams
            # prune beamsize
            beams.sort(key=lambda x: x.score, reverse=True)
            beams = beams[:self.beamsize]
            # prune beamthreshold
            best_score = beams[0].score
            beams = [b for b in beams if b.score >= best_score - self.beamthreshold]
            if t % 1 == 0:
                print(f" step {t} best:   {beams[0]}")
            for b in beams:
                print(f" {b} -> {self.shorten(b.text())}")

        # TODO: need a finishing step where I discard beams that are in the
        # middle of parsing a word.
        return beams

    def greedy_decode(self, x: np.array) -> str:
        idx = [np.argmax(step) for step in x]
        return ''.join([self.tokens[i] for i in idx])

    def shorten(self, text: str) -> str:
        out = ''.join([c for i, c in enumerate(text) if i == 0 or c != text[i-1]])
        return out.replace('#', '').replace('|', ' ')

    def fake_encode(self, s: str) -> np.array:
        """Generate a fake emissions matrix"""
        # add CTC blanks '#' where necessary
        tmp: list[str] = []
        for i, c in enumerate(s):
            tmp.append(c)
            if i < len(s)-1 and c == s[i+1]:
                tmp.append('#')
        s = ''.join(tmp)

        # use '|' to represent word breaks
        s = s.replace(' ', '|')

        out = np.zeros((0, len(self.tokens)))
        for c in s:
            width = random.randint(3, 5) if c != '#' else 1
            for i in range(width):
                x = np.random.rand(1, len(self.tokens)) * 9.99
                x[0,self.tokens.index(c)] = 10
                out = np.concatenate([out, x])
        return out

    def synthetic_test(self, s: str) -> None:
        self.test(self.fake_encode(s))

    def test(self, x: np.array) -> None:
        print('greedy(short)',  self.shorten(self.greedy_decode(x)))
        print('greedy()     ',  self.greedy_decode(x))
        beams = self.decode(x)
        if beams: print('decode()     ',  self.shorten(beams[0].text()))
        else: print('decode() FAILED!')
        for candidate in beams:
            print(f"{candidate.score:.3f} {candidate.text()}")


# ---------- TRIE LOADING ----------
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

def smear_score(node):
    """calculate the score for a try node"""
    # It is strange to mix log-add and max this way. It's what the original
    # does in SmearMode::MAX, but this may be a mistake. However, since we
    # effectively only use log_add if there's more than one word for a given
    # trie node, it's basically irrelevant for conformer.
    node.score = -math.inf
    for _, score in node.words:
        node.score = log_add(node.score, score)
    for child in node.edges.values():
        smear_score(child)
        node.score = max(node.score, child.score)

def load_trie(model: kenlm.Model, lexicon_path) -> TryNode:
    print("loading trie")
    root = TryNode(score=0, words=[], edges={})
    with open(lexicon_path, 'r') as f:
        for i, line in enumerate(f):
            word, *tokens = line.split()
            if (i+1) % 5000 == 0: print(f" {word}...", end='', flush=True)
            # bos=False means don't assume we're at beginning of sentence
            score = model.score(word, bos=False)
            node = root
            for t in tokens:
                if t not in node.edges:
                    node.edges[t] = TryNode(score=0, words=[], edges={})
                node = node.edges[t]
            node.words.append((word, score))

    print("\ncalculating scores...")
    smear_score(root)

    # just to get an idea of what some trie nodes look like
    for word in ["this", "lexicons", "airport"]:
        node = root
        for char in word: node = node.edges[char]
        print(f" {word} -> {node}")

    return root


# ---------- TESTING ----------
def main():
    tokens = "|'abcdefghijklmnopqrstuvwxyz"
    model = kenlm.Model(os.path.join(CONFORMER_PATH, "lm-ngram.bin"))
    trie = load_trie(model, os.path.join(CONFORMER_PATH, "lexicon.txt"))
    decoder = Decoder(tokens, beamsize=20, model=model, trie=trie)
    decoder.synthetic_test('a ')
    decoder.synthetic_test('hello ')
    decoder.synthetic_test('hello world ')
    decoder.synthetic_test('this is a decoder test ')

    print("---------- REAL DATA ----------")
    for n in "a hello hello_world this_is_a_decoder_test".split():
        filename = f'emit_tests/emit_{n}.npy'
        data = np.load(filename)
        print(f'\n# {filename}')
        decoder.test(data)

if __name__ == '__main__': main()
