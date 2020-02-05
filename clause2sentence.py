import re
import pandas as pd
from operator import itemgetter, attrgetter

firstConvertion = {
    "goal": "objectives",
    "hypothesis": "objectives",
    "fact": "background",
    "problem": "background",
    "method": "methods",
    "result": "results",
    "implication": "conclusion",
    "none":"background"
}


def hierarchicalDecision(labels):
    hierarchy = ["conclusion", "results", "methods", "objectives", "background"]
    for answer in hierarchy:
        if answer in labels:
            return answer
    assert(0)
    
def mergeFigureApprearance(figures):
    deduplicated = list(set(figures))
    if len(deduplicated)>1:
        no_nan = []
        for item in deduplicated:
            if type(item)==str:
                no_nan.extend(item.split("|"))
        deduplicated = ["|".join(list(set(no_nan)))]
        
    return deduplicated[0]

def readSeqLabelFigure(file_name):
    sentenceNums = []
    sentenceNum = []
    str_seqs = []
    str_seq = []
    label_seqs = []
    label_seq = []
    figure_spans = []
    figure_span = []
    figure_appearance = []
    figure_appearances = []
    df = pd.read_csv(file_name, sep='\t', header=0, index_col=0,engine='python')
    num_rec = df.shape[0]
    prev_paragraph = ""
    for i in range(num_rec):
        if type(df["Headings"][i]) == str:
            isResult = "result" in df["Headings"][i].lower()
        else:
            isResult = False
        if df["Paragraph"][i][0]=="p" and isResult: # e.g. "p1"
            if df["Paragraph"][i]!=prev_paragraph:
                prev_paragraph = df["Paragraph"][i]
                if len(str_seq)>0:
                    str_seqs.append(str_seq)
                    label_seqs.append(label_seq)
                    figure_spans.append(figure_span)
                    figure_appearances.append(figure_appearance)
                    sentenceNums.append(sentenceNum)
                str_seq = []
                label_seq = []
                figure_span = []
                figure_appearance = []
                sentenceNum = []
            str_seq.append(df["Clause Text"][i].lower()) # Lower case!
            label_seq.append(df["Discourse Type"][i].strip())
            figure_span.append(df["fig_spans"][i])
            figure_appearance.append(df["ExperimentValues"][i])
            sentenceNum.append(df["SentenceId"][i])
    return str_seqs, label_seqs, figure_spans, figure_appearances, sentenceNums

def clause2sentence(str_seqs, label_seqs, figure_spans, figure_appearances, all_sentenceNums):
    sentence_seqs = []
    new_label_seqs = []
    new_figure_spans = []
    new_figure_appearances = []
    for str_seq, label_seq, figure_span, figure_appearance, sentenceNums in zip(str_seqs, label_seqs, figure_spans, figure_appearances, all_sentenceNums):
        sentence_seq = []
        new_label_seq = []
        new_figure_span = []
        new_figure_appearance = []
        prev_sentence = ""
        for clause, label, figure, appearance, sentenceNum in zip(str_seq, label_seq, figure_span, figure_appearance, sentenceNums):
            if prev_sentence != sentenceNum:
                if prev_sentence!="":
                    sentence_seq.append(sentence)
                    RCT_labels = [firstConvertion[label] for label in labels]
                    new_label_seq.append(hierarchicalDecision(RCT_labels))
                    new_figure_span.append(mergeFigureApprearance(figures))
                    new_figure_appearance.append(mergeFigureApprearance(figure_appearance))
                    #print(sentence, figure_appearance, mergeFigureApprearance(figure_appearance))
                sentence = ""
                labels = []
                figures = []
                figure_appearance = []
            clause = clause.strip()
            sentence+=clause+" "
            labels.append(label)
            figures.append(figure)
            figure_appearance.append(appearance)
            tokens = clause.split(" ")
            prev_sentence = sentenceNum
            
        sentence_seq.append(sentence)
        RCT_labels = [firstConvertion[label] for label in labels]
        new_label_seq.append(hierarchicalDecision(RCT_labels))
        new_figure_span.append(mergeFigureApprearance(figures))
        new_figure_appearance.append(mergeFigureApprearance(figure_appearance))
        
        sentence_seqs.append(sentence_seq)
        new_label_seqs.append(new_label_seq)
        new_figure_spans.append(new_figure_span)
        new_figure_appearances.append(new_figure_appearance)
    return sentence_seqs, new_label_seqs, new_figure_spans, new_figure_appearances