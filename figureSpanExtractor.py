import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
from operator import itemgetter, attrgetter

def extractFigureMention(sentence):
    next_start = 0
    figureMentions = []
    while True:
        found = _extractFigureMention(sentence[next_start:])
        if len(found)>0:
            figureMentions.append(found[0])
            next_start = next_start + sentence[next_start:].index(found[0])+skipFigure(found[0])
        else:
            break
    return list(set(figureMentions))

def _extractFigureMention(sentence):
    def cutParenthesis(result):
        tokens = result.split()
        filtered = []
        for token in tokens:
            if token==")":
                break
            elif len(token)==1 or token[0].isdigit() or "-" in token or "fig" in token or "Fig" in token or "and" in token:
                filtered.append(token)
            else:
                break
        result = " ".join(filtered)
        return result
    results = re.findall(r'(?:[Ff]igs?.|[Ff]igures?)[\s?,?\s?(?:\d+\w?|\w\s?\-\s?\w)\s?]+', sentence)
    results = [cutParenthesis(result) for result in results]
    return results

def findBetween(char1, char2):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    start = alphabet.index(char1)
    end = alphabet.index(char2)
    return alphabet[start:end+1]

def extractNumber(inputString):
    result = re.findall(r'\d+', inputString)
    if len(result)>0:
        return result[0]
    else:
        return None
    
def extractRawFigSpan(extracted):
    """
    Given a list of extracted strings that mention figure numbers, extract figure numbers.
    """
    figures = []
    for fragment in extracted:
        tokens = word_tokenize(fragment.lower())
        for i, token in enumerate(tokens):
            if token==")":
                tokens = tokens[:i]
                break
        
        prev_fig_number_used = True
        for token in tokens:
            new_fig_number = extractNumber(token)
            if bool(re.search(r'(?:[Ff]igs?|[Ff]igures?)', token)) or token=="and" or token=="&" or token=="compare":
                pass
            elif new_fig_number:
                if not prev_fig_number_used:
                    figures.append(fig_number) # When inputs are like fig. 8 , 9 without sub-figures
                fig_number = new_fig_number
                prev_fig_number_used = False
                if bool(re.search(r'\d+\w', token)):
                    figures.append(token)
                    prev_fig_number_used = True
            elif bool(re.search(r'\w', token)):
                if len(token)==1 or "-" in token:
                    figures.append(fig_number+token)
                    prev_fig_number_used = True
                else:
                    break
            elif "-" in token:
                figures.append(token)
            elif token!="," and token!=".":
                #print(token)
                break
        if not prev_fig_number_used:
            figures.append(fig_number) # When inputs are like fig. 8 without sub-figures
    return figures

def refineFigureSpan(figures):
    """
    Given a list of extracted figure numbers, expand - to full figure numbers.
    E.g. converting 3a-d to 3a,3b,3c,3d.
    """
    final_figure = []
    normal_count = 0
    expansion_count = 0
    for i, token in enumerate(figures):
        if token=="-":
            expansion_count += 1
            figure_num = extractNumber(figures[i-1])
            start = re.findall(r'\w', figures[i-1])[0]
            end = re.findall(r'\w', figures[i+1])[0]
            full_letters = findBetween(start, end)
            for letter in full_letters:
                final_figure.append(figure_num+letter)
        elif "-" in token:
            expansion_count += 1
            figure_num = extractNumber(token)
            pivot = token.index("-")
            start = token[pivot-1]
            end = token[pivot+1]
            full_letters = findBetween(start, end)
            for letter in full_letters:
                final_figure.append(figure_num+letter)
        else:
            normal_count += 1
            final_figure.append(token)
    final_figure = sorted(list(set(final_figure)))
    return final_figure

def skipFigure(fragment):
    result = re.findall(r'(?:[Ff]igs?.?|[Ff]igures?)', fragment)
    return len(result)

def extractFigureSpan(sentence):
    """
    Wrapper of the rule-based figure span extractor.
    """
    figureMentions = extractFigureMention(sentence)
        
    #print(figureMentions)
    figure_span = extractRawFigSpan(figureMentions)
    #print(raw_figure_span)
    #figure_span = refineFigureSpan(raw_figure_span)
    return figure_span

