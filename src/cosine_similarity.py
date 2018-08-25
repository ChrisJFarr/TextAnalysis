from re import compile
from math import sqrt
from collections import Counter
from nameparser import HumanName
# http://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
# General class used for measuring cosine similarity of text strings


class CosineSimilarity:

    def __init__(self):
        self.__word = compile(r'\w+')

    @staticmethod
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = sqrt(sum1) * sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(self, text):
        words = self.__word.findall(text)
        return Counter(words)

    def cosinecalc(self, text1, text2):
        if len(text1) < 1 or len(text2) < 1:
            return "Document missing"
        else:
            vector1 = self.text_to_vector(text1)
            vector2 = self.text_to_vector(text2)
            cosine = self.get_cosine(vector1, vector2)
            return cosine

    @staticmethod
    def names_compare(name1, name2):
        """
        Takes string arguments with human names and returns indication of match between names.
        :param name1: String argument with name
        :param name2: String argument with name
        :return: "exact" for definite matches, "last" for only last name matches, False for non-matches
        """

        if not isinstance(name1, str) or not isinstance(name2, str):
            raise TypeError("CosineCalc.names_compare must receive both string arguments.")

        name1 = HumanName(name1.lower())
        name2 = HumanName(name2.lower())

        # Check for exact matches
        last_names_match = name1.last == name2.last
        first_names_match = name1.first == name2.first

        # Check for short names
        order = len(name1.first) < len(name2.first)
        if order:
            nick_name = name1.first in name2.first
        else:
            nick_name = name2.first in name1.first

        if last_names_match and first_names_match:
            result = "exact"
        elif last_names_match and nick_name:
            result = "exact"
        elif last_names_match:
            result = "last"
        else:
            result = False

        return result


