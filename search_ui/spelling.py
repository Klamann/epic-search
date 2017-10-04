from enum import Enum
from typing import List, Tuple

import editdistance
from hunspell import Hunspell


class Spelling(Enum):
    """
    the status of a token that has been processed by the spell checker.
    a token can be
    - correct: no change required
    - fixed: the token was misspelled and a correction was found
    - failed: the token is misspelled and no correction was found
    """
    correct = 1
    fixed = 2
    failed = 3


class SpellChecker(Hunspell):

    def __init__(self, *args, encoding='utf-8', **kwargs):
        super().__init__(*args, **kwargs)
        if encoding:
            self._dic_encoding = encoding

    def suggest(self, word: str, max_distance: int = None):
        suggestions = super().suggest(word)
        if max_distance:
            return [suggestion for suggestion in suggestions
                    if editdistance.eval(word, suggestion) <= max_distance]
        else:
            return suggestions

    def spell(self, word: str) -> bool:
        return super().spell(word)

    def correct(self, text: str, max_distance: int = None) -> List[Tuple[str, Spelling]]:
        """
        checks the spelling in the text.
        returns the tokens that this text consists of, where each token that is not
        correct has been replaced by it's best correct counterpart, if possible.
        :param text: the text to correct
        :param max_distance: the maximum allowed Levenshtein distance for corrections
        :return: a list of tuples (token, status), where status indicates whether a token
                  was correct or misspelled and whether a replacement was found
        """
        corrected_tokens = []
        for token in text.split():
            correct = self.spell(token)
            if correct:
                corrected_tokens.append((token, Spelling.correct))
            else:
                suggestions = self.suggest(token, max_distance=max_distance)
                # do not allow suggestions that consist of more than one word
                suggestions_filtered = [s for s in suggestions if s.isalnum()]
                if suggestions_filtered:
                    corrected_tokens.append((suggestions_filtered[0], Spelling.fixed))
                else:
                    corrected_tokens.append((token, Spelling.failed))
        return corrected_tokens
