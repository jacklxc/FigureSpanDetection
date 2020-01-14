import codecs
import numpy as np
import glob
import re
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from clause2sentence import readSeqLabelFigure, clause2sentence
from glob import glob
from figureSpanExtractor import extractFigureSpan, extractFigureMention

def getSentenceCharRep(sentence, max_char_len):
    tokens = sentence.split()
    reps = np.zeros((len(tokens), max_char_len))
    for wi, word in enumerate(tokens):
        for ci, char in enumerate(word):
            reps[wi,ci] = ord(char)
    return reps

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.astype(int)])

def to_BIO_simple(boolean_array):
    BIO = []
    prev = False
    for element in boolean_array:
        if element and prev:
            BIO.append("I")
        elif element and not prev:
            BIO.append("B")
            prev = True
        else:
            BIO.append("O")
            prev = False
    return BIO

def cleanFigureAnnotation(all_figures, placeholder = None):
    processed_all_figures = []
    for para in all_figures:
        processed_paragraph_figures = []
        for raw_figures in para:
            processed_sentence_figures = []
            if type(raw_figures)==str:
                figures = raw_figures.split("|")
                for figure in figures:
                    if figure[0]=="f":
                        processed_sentence_figures.append(figure[1:])
                    else:
                        processed_sentence_figures.append(figure)
            elif type(raw_figures) == float and placeholder is not None:
                processed_sentence_figures.append(placeholder)
            processed_paragraph_figures.append(processed_sentence_figures)
        processed_all_figures.append(processed_paragraph_figures)
    return processed_all_figures

def candidatesPerParagraph(processed_all_figures):
    all_candidates = []
    for para in processed_all_figures:
        candidates = set([])
        for figures in para:
            for figure in figures:
                candidates.add(figure)
        candidates = sorted(list(candidates))
        all_candidates.append(candidates)
    return all_candidates

def makeCandidateLookup(all_candidates):
    all_indices = {}
    index = 0
    for candidates in all_candidates:
        for candidate in candidates:
            if candidate not in all_indices:
                all_indices[candidate] = index
                index += 1
    return all_indices

def makeFigureSpanBitmap(all_figure_spans, candidate_lookup, maxParagraphLen):
    """
    (N_paragraph, N_figure_considered, maxParagraphLen)
    A condensed representation of the output. Feed each bitmap[:,f1,:] as part of training label.
    """
    bitmap = np.zeros((len(all_figure_spans), len(candidate_lookup), maxParagraphLen))
    for pi, figure_span_para in enumerate(all_figure_spans):
        for si, figure_span in enumerate(figure_span_para):
            for figure in figure_span:
                bitmap[pi, candidate_lookup[figure],si] = 1
    return bitmap

def makeFigureCharEmbedding(figure, figureEmbeddingDim=3):
    """
    Return shape (figureEmbeddingDim,)
    """
    figure_embedding = []
    to_return = np.zeros((figureEmbeddingDim,))
    for char in figure:
        figure_embedding.append(ord(char))
    char_len = len(figure_embedding)
    truncate = min(char_len, figureEmbeddingDim)
    to_return[:truncate] = np.array(figure_embedding[:truncate])
    return to_return

def makeCandidateCharEmbedding(all_candidates, maxFigureNum, figureEmbeddingDim=3):
    numParagraph = len(all_candidates)
    X = np.zeros((numParagraph, maxFigureNum, figureEmbeddingDim))
    for pi, paragraph in enumerate(all_candidates):
        for fi, figure in enumerate(paragraph):
            X[pi, fi, :] = makeFigureCharEmbedding(figure, figureEmbeddingDim)
    return X

def makeAppearanceCharEmbedding(all_appearances, maxseqlen, maxFigureNum, figureEmbeddingDim=3):
    numParagraph = len(all_appearances)
    X = np.zeros((numParagraph, maxseqlen, maxFigureNum, figureEmbeddingDim))
    for pi, paragraph in enumerate(all_appearances):
        for si, sentence in enumerate(paragraph):
            for fi, figure in enumerate(sentence):
                X[pi,si,fi,:] = makeFigureCharEmbedding(figure, figureEmbeddingDim)
    return X
    
def readAllSeqLabelFigure(folderpath):
    file_paths = glob(folderpath+"*.tsv")
    all_texts = []
    all_labels = []
    all_figures = []
    all_figure_appearance = []
    str_seq_lens = []
    for file_name in file_paths:
        str_seqs, label_seqs, figure_spans, figure_appearances, sentenceNums = readSeqLabelFigure(file_name)
        #str_seqs, label_seqs, figure_spans, figure_appearances = clause2sentence(str_seqs, label_seqs, figure_spans, figure_appearances, sentenceNums)
        all_texts.extend(str_seqs)
        all_labels.extend(label_seqs)
        all_figures.extend(figure_spans)
        all_figure_appearance.extend(figure_appearances)
        str_seq_lens.append(len(str_seqs))
        print(file_name, len(str_seqs), len(all_texts))
    return all_texts, all_labels, all_figures, all_figure_appearance, str_seq_lens

def read_passages(filename, is_labeled):
    str_seqs = []
    str_seq = []
    label_seqs = []
    label_seq = []
    for line in codecs.open(filename, "r", "utf-8"):
        lnstrp = line.strip()
        if lnstrp == "":
            if len(str_seq) != 0:
                str_seqs.append(str_seq)
                str_seq = []
                label_seqs.append(label_seq)
                label_seq = []
        else:
            if is_labeled:
                clause, label = lnstrp.split("\t")
                label_seq.append(label.strip())
            else:
                clause = lnstrp
            str_seq.append(clause)
    if len(str_seq) != 0:
        str_seqs.append(str_seq)
        str_seq = []
        label_seqs.append(label_seq)
        label_seq = []
    return str_seqs, label_seqs

def from_BIO_ind(BIO_pred, BIO_target, indices):
    table = {} # Make a mapping between the indices of BIO_labels and temporary original label indices
    original_labels = []
    for BIO_label,BIO_index in indices.items():
        if BIO_label[:2] == "I_" or BIO_label[:2] == "B_":
            label = BIO_label[2:]
        else:
            label = BIO_label
        if label in original_labels:
            table[BIO_index] = original_labels.index(label)
        else:
            table[BIO_index] = len(original_labels)
            original_labels.append(label)

    original_pred = [table[label] for label in BIO_pred]
    original_target = [table[label] for label in BIO_target]
    return original_pred, original_target

def to_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        prev = ""
        for label in label_para:
            if label!="none": # "none" is O, remain unchanged.
                if label==prev:
                    new_label = "I_"+label
                else:
                    new_label = "B_"+label
            else:
                new_label = label # "none"
            prev = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def from_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        for label in label_para:
            if label[:2] == "I_" or label[:2] == "B_":
                new_label = label[2:]
            else:
                new_label = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def clean_url(word):
    """
        Clean specific data format from social media
    """
    # clean urls
    word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
    word = re.sub(r'exlink', '<URL>', word)
    return word

def clean_num(word):
    # check if the word contain number and no letters
    if any(char.isdigit() for char in word):
        try:
            num = float(word.replace(',', ''))
            return '@'
        except:
            if not any(char.isalpha() for char in word):
                return '@'
    return word


def clean_words(str_seqs):
    processed_seqs = []
    for str_seq in str_seqs:
        processed_clauses = []
        for clause in str_seq:
            filtered = []
            tokens = clause.split()                 
            for word in tokens:
                word = clean_url(word)
                word = clean_num(word)
                filtered.append(word)
            filtered_clause = " ".join(filtered)
            processed_clauses.append(filtered_clause)
        processed_seqs.append(processed_clauses)
    return processed_seqs

def test_f1(gold_label_seqs,pred_label_seqs):
    def linearize(labels):
        linearized = []
        for label in labels:
            linearized.extend(label)
        return linearized
    true_label = linearize(gold_label_seqs)
    pred_label = linearize(pred_label_seqs)

    f1 = f1_score(true_label,pred_label,average="weighted")
    print("Sequence F1 score:",f1)
    return f1
    
def evaluate(y, pred):
    accuracy = float(sum([c == p for c, p in zip(y, pred)]))/len(pred)
    num_gold = {}
    num_pred = {}
    num_correct = {}
    for c, p in zip(y, pred):
        if c in num_gold:
            num_gold[c] += 1
        else:
            num_gold[c] = 1
        if p in num_pred:
            num_pred[p] += 1
        else:
            num_pred[p] = 1
        if c == p:
            if c in num_correct:
                num_correct[c] += 1
            else:
                num_correct[c] = 1
    fscores = {}
    for p in num_pred:
        precision = float(num_correct[p]) / num_pred[p] if p in num_correct else 0.0
        recall = float(num_correct[p]) / num_gold[p] if p in num_correct else 0.0
        fscores[p] = 2 * precision * recall / (precision + recall) if precision !=0 and recall !=0 else 0.0
    weighted_fscore = sum([fscores[p] * num_gold[p] if p in num_gold else 0.0 for p in fscores]) / sum(num_gold.values())
    return accuracy, weighted_fscore, fscores

def make_folds(train_X, train_Y, num_folds):
    num_points = train_X.shape[0]
    fol_len = num_points / num_folds
    rem = num_points % num_folds
    print(train_X.shape, train_Y.shape)
    X_folds = np.split(train_X, num_folds) if rem == 0 else np.split(train_X[:-rem], num_folds)
    Y_folds = np.split(train_Y, num_folds) if rem == 0 else np.split(train_Y[:-rem], num_folds)
    cv_folds = []
    for i in range(num_folds):
        train_folds_X = []
        train_folds_Y = []
        for j in range(num_folds):
            if i != j:
                train_folds_X.append(X_folds[j])
                train_folds_Y.append(Y_folds[j])
        train_fold_X = np.concatenate(train_folds_X)
        train_fold_Y = np.concatenate(train_folds_Y)
        cv_folds.append(((train_fold_X, train_fold_Y), (X_folds[i], Y_folds[i])))
    return cv_folds

def make_folds_multi(train_X, train_Y, num_folds):
    for key, v in train_X.items():
        num_points = train_X[key].shape[0]
        break
    fol_len = num_points / num_folds
    rem = num_points % num_folds
    all_X_folds = {}
    all_Y_folds = {}
    for key, X in train_X.items():
        X_folds = np.split(X, num_folds) if rem == 0 else np.split(X[:-rem], num_folds)
        all_X_folds[key] = X_folds
    for key, Y in train_Y.items():
        Y_folds = np.split(Y, num_folds) if rem == 0 else np.split(Y[:-rem], num_folds)
        all_Y_folds[key] = Y_folds

    cv_folds = []
    for i in range(num_folds):
        train_folds_X = {key:[] for key,X in train_X.items()}
        train_folds_Y = {key:[] for key,Y in train_Y.items()}
        for j in range(num_folds):
            if i != j:
                for key, X_folds in all_X_folds.items():
                    train_folds_X[key].append(all_X_folds[key][j])
                for key, Y_folds in all_Y_folds.items():
                    train_folds_Y[key].append(all_Y_folds[key][j])
        
        train_fold_X = {key: np.concatenate(X) for key, X in train_folds_X.items()}
        train_fold_Y = {key: np.concatenate(Y) for key, Y in train_folds_Y.items()}
        test_fold_X = {key: X[i] for key, X in all_X_folds.items()}
        test_fold_Y = {key: Y[i] for key, Y in all_Y_folds.items()}
        cv_folds.append(((train_fold_X, train_fold_Y), (test_fold_X,test_fold_Y)))
    return cv_folds

def arg2param(args):
    params = vars(args)
    params["lr"] = float(args.lr)
    params["hard_k"] = int(args.hard_k)
    params["discourse_embedding_dropout"] = float(args.discourse_embedding_dropout)
    params["discourse_high_dense_dropout"] = float(args.discourse_high_dense_dropout)
    params["discourse_attention_dropout"] = float(args.discourse_attention_dropout)
    params["discourse_lstm_dropout"] = float(args.discourse_lstm_dropout)
    params["discourse_word_proj_dim"] = int(args.discourse_word_proj_dim)
    params["discourse_lstm_dim"] = int(args.discourse_lstm_dim)
    params["discourse_att_proj_dim"] = int(args.discourse_att_proj_dim)
    params["discourse_rec_hid_dim"] = int(args.discourse_rec_hid_dim)
    params["embedding_dropout"] = float(args.embedding_dropout)
    params["high_dense_dropout"] = float(args.high_dense_dropout)
    params["attention_dropout"] = float(args.attention_dropout)
    params["lstm_dropout"] = float(args.lstm_dropout)
    params["word_proj_dim"] = int(args.word_proj_dim)
    params["lstm_dim"] = int(args.lstm_dim)
    params["att_proj_dim"] = int(args.att_proj_dim)
    params["rec_hid_dim"] = int(args.rec_hid_dim)
    params["epoch"] = int(args.epoch)
    if args.maxseqlen is not None:
        params["maxseqlen"] = int(args.maxseqlen)
    if args.maxclauselen is not None:
        params["maxclauselen"] = int(args.maxclauselen)
    params["batch_size"]=int(args.batch_size)
    params["validation_split"] = float(args.validation_split)
    params["figure_proj_dim"] = int(args.figure_proj_dim)
    params["figure_rnn_dim"] = int(args.figure_rnn_dim)
    params["figure_lstm_dim"] = int(args.figure_lstm_dim)
    params["figure_embedding_dropout"] = float(args.figure_embedding_dropout)
    params["figure_high_dense_dropout"] = float(args.figure_high_dense_dropout)
    params["figure_rnn_dropout"] = float(args.figure_rnn_dropout)
    params["figure_lstm_dropout"] = float(args.figure_lstm_dropout)
    return params

def flatten(listOfList):
    toReturn = []
    for List in listOfList:
        toReturn.extend(List)
    return toReturn

def blockBIO(paragraphs, placeholder = "NaN", mapping=None):
    all_BIO = []
    for para_label in paragraphs:
        para_BIO = []
        prev = ""
        for label in para_label:
            if label==[placeholder] or type(label)==float:
                if mapping is not None:
                    para_BIO.append(mapping["O"])
                else:
                    para_BIO.append("O")
            elif label!=prev:
                if mapping is not None:
                    para_BIO.append(mapping["B"])
                else:
                    para_BIO.append("B")
            else:
                if mapping is not None:
                    para_BIO.append(mapping["I"])
                else:
                    para_BIO.append("I")
            prev = label
        all_BIO.append(para_BIO)
    return all_BIO

def sortFigureAnnotation(paragraphs, placeholder=None):
    all_figures = []
    for para_label in paragraphs:
        para_figure = []
        for label in para_label:
            if type(label)==str:
                processed = "|".join(sorted(label.split("|")))
            else:
                if placeholder is not None:
                    processed = placeholder
                else:
                    processed = label
            para_figure.append(processed)
        all_figures.append(para_figure)
    return all_figures

def BIO2FigureLabel(all_BIO, figure_appearances, placeholder="NaN"):
    all_figure = []
    for para_BIO, para_appearance in zip(all_BIO, figure_appearances):
        para_figure = []
        appeared = []
        prev_appeared = []
        buffer_count = 0
        for BIO, appearance in zip(para_BIO, para_appearance):
            if BIO=="O" or BIO=="NA":
                if buffer_count > 0:
                    processed_appeared = sorted(list(set(appeared)))
                    if len(processed_appeared)==0:
                        processed_appeared = prev_appeared
                    for i in range(buffer_count):
                        para_figure.append(processed_appeared)
                    buffer_count = 0
                    prev_appeared = processed_appeared
                    appeared = []
                para_figure.append([placeholder]) # If appearance is not NaN, should put whatever appeared
            elif BIO=="B":
                if buffer_count > 0:
                    processed_appeared = sorted(list(set(appeared)))
                    if len(processed_appeared)==0:
                        processed_appeared = prev_appeared
                    for i in range(buffer_count):
                        para_figure.append(processed_appeared)
                    buffer_count = 0
                    prev_appeared = processed_appeared
                    appeared = []
                buffer_count += 1
                if appearance != [placeholder]:
                    appeared.extend(appearance)
            elif BIO=="I":
                buffer_count += 1
                if appearance != [placeholder]:
                    appeared.extend(appearance)
            else:
                assert(0)
        if buffer_count > 0:
            processed_appeared = sorted(list(set(appeared)))
            if len(processed_appeared)==0:
                processed_appeared = prev_appeared
            for i in range(buffer_count):
                para_figure.append(processed_appeared)
            buffer_count = 0
            prev_appeared = processed_appeared
            appeared = []
        
        # Put placeholder to empty lists
        for sent_figure in para_figure:
            if len(sent_figure)==0:
                sent_figure.append(placeholder)
        all_figure.append(para_figure)
        
    return all_figure

def excludePlaceholder(gold, preds, placeholder="NaN"):
    filtered_gold = []
    filtered_preds = []
    for true_para, pred_para in zip(gold, preds):
        filtered_true_para = []
        filtered_pred_para = []
        for true_sentence, pred_sentence in zip(true_para, pred_para):
            if true_sentence != [placeholder]:
                filtered_true_para.append(true_sentence)
                filtered_pred_para.append(pred_sentence)
        if len(filtered_true_para) > 0:
            filtered_gold.append(filtered_true_para)
            filtered_preds.append(filtered_pred_para)
    return filtered_gold, filtered_preds

def computeF1(cleanedFigure, predictions):
    support = []
    f1s = []
    for true, pred in zip(cleanedFigure, predictions):
        support.append(len(true))
        binarizer = MultiLabelBinarizer().fit(true+pred)
        y_true = binarizer.transform(true)
        y_pred = binarizer.transform(pred)
        f1s.append(f1_score(y_true, y_pred, average='micro'))
    F1 = np.average(f1s,weights=support)
    return F1

def getNumAlphabetMapping(figures, placeholder="NaN"):
    if placeholder in figures:
        figures.remove(placeholder)
    to_replace = list(set("".join(figures)))
    mapping = {}
    num_count = 0
    alphabet_count = 0
    for char in to_replace:
        if char.isdigit():
            mapping[char] = "<NUM"+str(num_count)+">"
            num_count+=1
        elif char.isalpha():
            mapping[char] = "<CHAR"+str(alphabet_count)+">"
            alphabet_count+=1
    return mapping

def replaceNumAlphabet(mentions, mapping):
    replaced_mentions = []
    for mention in mentions:
        tokens = mention.split()
        replaced_tokens = []
        for token in tokens:
            if len(token)==1:
                replaced_token = mapping.get(token,token)
            else:
                replaced_token = token
            replaced_tokens.append(replaced_token)
        replaced_mentions.append(" ".join(replaced_tokens))
    return replaced_mentions

def replaceFigureSeqNumAlphabet(figure_seq, mappings, placeholder="NaN"):
    replaced_figure_seq = []
    for sentence_figure, mapping in zip(figure_seq, mappings):
        replaced_figure_sentence = []
        for figure in sentence_figure:
            if figure != placeholder:
                replaced_figure = [mapping[char] for char in figure]
                replaced_figure_sentence.append(" ".join(replaced_figure))
            else:
                replaced_figure_sentence.append(figure)
        replaced_figure_seq.append(replaced_figure_sentence)
    return replaced_figure_seq

def maskFigureNum(str_seqs, figure_seqs, figure_appearances, candidate_seqs, placeholder="NaN"):
    replaced_str_seqs = []
    mappings = []
    replaced_figure_seqs = []
    replaced_figure_appearances = []
    for str_seq, figure_seq, figure_appearance in zip(str_seqs, figure_seqs, figure_appearances):
        mentions = []
        possible_figures = flatten(figure_seq)
        possible_figures = sorted(list(set(possible_figures)))
        for sentence in str_seq:
            mentions.extend(extractFigureMention(sentence))
        mentions = sorted(list(set(mentions)))
        mapping = getNumAlphabetMapping(possible_figures)
        replaced_mentions = replaceNumAlphabet(mentions,mapping)

        replaced_sentences = []
        for sentence in str_seq:
            for mention, replaced in zip(mentions, replaced_mentions):
                sentence = sentence.replace(mention, replaced)
            replaced_sentences.append(sentence)
        replaced_figure_seq = replaceFigureSeqNumAlphabet(figure_seq, [mapping for i in range(len(figure_seq))], placeholder)
        replaced_figure_appearance = replaceFigureSeqNumAlphabet(figure_appearance, [mapping for i in range(len(figure_appearance))], placeholder)

        replaced_str_seqs.append(replaced_sentences)
        mappings.append(mapping)
        replaced_figure_seqs.append(replaced_figure_seq)
        replaced_figure_appearances.append(replaced_figure_appearance)
    replaced_candidates = None
    #replaced_candidates = replaceFigureSeqNumAlphabet(candidate_seqs, mappings)
    return replaced_str_seqs, replaced_figure_seqs, replaced_figure_appearances,replaced_candidates, mappings


def blockBIO_extended(figure_seqs, figure_appearances, placeholder = "NaN"):
    figure_appearances = deepcopy(figure_appearances)
    def remove_placeholder(block):
        while placeholder in block:
            block.remove(placeholder)
        return block
            
    all_BIO = blockBIO(figure_seqs, placeholder=placeholder)
    all_extended_BIO = []
    for para_label, para_appearances, para_BIO in zip(figure_seqs, figure_appearances, all_BIO):
        # Prepare labels, appearances and BIO by grouping them into blocks for later process.
        para_extended_BIO = []
        figure_seq_blocks = []
        appearance_blocks = []
        appearance_sentence_blocks = []
        block_indices = []
        this_seq_block = []
        for i, (labels, appearances, BIO) in enumerate(zip(para_label, para_appearances, para_BIO)):
            if BIO=="B":
                if len(this_seq_block) > 0: 
                    figure_seq_blocks.append(set(remove_placeholder(this_seq_block)))
                    appearance_blocks.append(set(remove_placeholder(this_appearance_block)))
                    appearance_sentence_blocks.append(this_appearance_sentence_block)
                    block_indices.append(block_index) 
                this_seq_block = []
                this_appearance_block = []
                this_appearance_sentence_block = []
                this_seq_block.extend(labels)
                this_appearance_block.extend(appearances)
                this_sentence_appearance = remove_placeholder(appearances)
                if len(this_sentence_appearance) > 0:
                    this_appearance_sentence_block.append(this_sentence_appearance)
                block_index = [i]
            elif BIO=="I":
                this_seq_block.extend(labels)
                this_appearance_block.extend(appearances)
                this_sentence_appearance = remove_placeholder(appearances)
                if len(this_sentence_appearance) > 0:
                    this_appearance_sentence_block.append(this_sentence_appearance)
                block_index.append(i)
            else:
                if len(this_seq_block) > 0: 
                    figure_seq_blocks.append(set(remove_placeholder(this_seq_block)))
                    appearance_blocks.append(set(remove_placeholder(this_appearance_block)))
                    appearance_sentence_blocks.append(this_appearance_sentence_block)
                    block_indices.append(block_index) 
                this_seq_block = []
                this_appearance_block = []
                figure_seq_blocks.append(set([]))
                appearance_blocks.append(set([]))
                appearance_sentence_blocks.append([])
                block_indices.append([i]) 
        
        if len(this_seq_block) > 0: 
            figure_seq_blocks.append(set(remove_placeholder(this_seq_block)))
            appearance_blocks.append(set(remove_placeholder(this_appearance_block)))
            appearance_sentence_blocks.append(this_appearance_sentence_block)
            block_indices.append(block_index)
        #print(figure_seq_blocks, appearance_blocks, appearance_sentence_blocks, block_indices)
        
        # Construct modified BIO sequence
        para_extended_BIO = []
        for i, (figure_seq_block, appearance_block, block_index) \
            in enumerate(zip(figure_seq_blocks, appearance_blocks, block_indices)):
            if len(figure_seq_block)==0:
                para_extended_BIO.append("O")
            elif figure_seq_block==appearance_block:
                para_extended_BIO.append("B")
                para_extended_BIO.extend(["I"]*(len(block_index)-1))
            elif len(figure_seq_block - appearance_block) > 0:
                if i==0: # No previous block
                    para_extended_BIO.append("B_down")
                    para_extended_BIO.extend(["I_down"]*(len(block_index)-1))
                elif i==len(figure_seq_blocks)-1:
                    para_extended_BIO.append("B_up")
                    para_extended_BIO.extend(["I_up"]*(len(block_index)-1))
                else: 
                    difference = figure_seq_block - appearance_block
                    previous_block = appearance_sentence_blocks[i-1]
                    next_block = appearance_sentence_blocks[i+1]
                    if len(previous_block)==0 and len(next_block) ==0: # Give up, very special case
                        para_extended_BIO.extend(["SP"]*len(block_index)) ###
                    elif len(previous_block)==0:
                        para_extended_BIO.append("B_down")
                        para_extended_BIO.extend(["I_down"]*(len(block_index)-1))
                    elif len(next_block) == 0:
                        para_extended_BIO.append("B_up")
                        para_extended_BIO.extend(["I_up"]*(len(block_index)-1))
                    elif len(difference - set(previous_block[-1])) == 0:
                        para_extended_BIO.append("B_up")
                        para_extended_BIO.extend(["I_up"]*(len(block_index)-1))
                    elif len(difference - set(next_block[0])) == 0:
                        para_extended_BIO.append("B_down")
                        para_extended_BIO.extend(["I_down"]*(len(block_index)-1))
                    else:
                        para_extended_BIO.append("B_both")
                        para_extended_BIO.extend(["I_both"]*(len(block_index)-1))
            else:
                print("Unexpected case occured!")
                para_extended_BIO.extend(["SP"]*len(block_index))
        #print(para_extended_BIO, para_BIO)
        assert(len(para_extended_BIO) == len(para_BIO))
        #print()
        all_extended_BIO.append(para_extended_BIO)
    return all_extended_BIO

def BIO2FigureLabel_extended(all_BIO, figure_appearances, placeholder="NaN"):
    all_figure = []
    for para_BIO, para_appearance in zip(all_BIO, figure_appearances):
        para_figure = []
        appeared = []
        prev_appeared = []
        buffer_count = 0
        appearance_block = []
        block_indices = []
        this_block_index = []

        for index, (BIO, appearance) in enumerate(zip(para_BIO, para_appearance)):
            if BIO=="O" or BIO=="NA":
                if buffer_count > 0:
                    processed_appeared = sorted(list(set(appeared)))
                    #if len(processed_appeared)==0:
                    #    processed_appeared = prev_appeared
                    for i in range(buffer_count):
                        para_figure.append(processed_appeared)
                    appearance_block.append(appearance_this_block)
                    block_indices.append(this_block_index)
                    # Initialization for next block
                    buffer_count = 0
                    prev_appeared = processed_appeared
                    appeared = []
                para_figure.append(appearance) # If appearance is not NaN, should put whatever appeared
                block_indices.append([index])
                if appearance!=[placeholder]:
                    appearance_block.append([appearance])
                else:
                    appearance_block.append([])
            elif BIO[0]=="B":
                if buffer_count > 0:
                    processed_appeared = sorted(list(set(appeared)))
                    #if len(processed_appeared)==0:
                    #    processed_appeared = prev_appeared
                    for i in range(buffer_count):
                        para_figure.append(processed_appeared)
                    appearance_block.append(appearance_this_block)
                    block_indices.append(this_block_index)
                    
                    # Initialization for next block
                    this_block_index = []
                    buffer_count = 0
                    prev_appeared = processed_appeared
                    appeared = []
                buffer_count += 1
                appearance_this_block = []
                this_block_index = [index]
                if appearance != [placeholder]:
                    appeared.extend(appearance)
                    appearance_this_block.append(appearance)
            elif BIO[0]=="I":
                buffer_count += 1
                this_block_index.append(index)
                if appearance != [placeholder]:
                    appeared.extend(appearance)
                    appearance_this_block.append(appearance)
            else:
                assert(0)
        if buffer_count > 0:
            processed_appeared = sorted(list(set(appeared)))
            #if len(processed_appeared)==0:
            #    processed_appeared = prev_appeared
            appearance_block.append(appearance_this_block)
            block_indices.append(this_block_index)
            for i in range(buffer_count):
                para_figure.append(processed_appeared)
            buffer_count = 0
            prev_appeared = processed_appeared
            appeared = []
        #print(block_indices, para_BIO)
        # Modify sequences for extended BIO labels
        #print(block_indices, appearance_block)
        for ib, indices in enumerate(block_indices):
            if "down" in para_BIO[indices[0]]:
                if ib < len(block_indices) - 1 and len(appearance_block[ib+1])>0:
                    for index in indices:
                        para_figure[index].extend(appearance_block[ib+1][0])
            
            elif "up" in para_BIO[indices[0]] and len(appearance_block[ib-1])>0:
                if ib > 0:
                    for index in indices:
                        para_figure[index].extend(appearance_block[ib-1][-1])
            elif "both" in para_BIO[indices[0]]:
                if ib < len(block_indices) - 1 and len(appearance_block[ib+1])>0:
                    for index in indices:
                        para_figure[index].extend(appearance_block[ib+1][0])
                if ib > 0 and len(appearance_block[ib-1])>0:
                    for index in indices:
                        para_figure[index].extend(appearance_block[ib-1][-1])
                        
        # Put placeholder to empty lists
        for sent_figure in para_figure:
            if len(sent_figure)==0:
                sent_figure.append(placeholder)
        all_figure.append(para_figure)
    return all_figure

def computePaperF1(figure_seqs, pred_figure, start=0, end=None):
    if end is None:
        end = len(figure_seqs)
    def flatten(paragraphs):
        out = []
        for paragraph in paragraphs:
            out.extend(paragraph)
        return out
    
    def figure_candidates(paper):
        candidates = set([])
        for sentence in paper:
            for figure in sentence:
                candidates.add(figure)
        return sorted(list(candidates))
    
    paper_gold = flatten(figure_seqs[start:end])
    paper_pred = flatten(pred_figure[start:end])
    paper_candidates = figure_candidates(paper_gold)
    
    # Compute F1
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for candidate in paper_candidates:
        for gold, pred in zip(paper_gold, paper_pred):
            inGold = candidate in gold
            inPred = candidate in pred
            if inGold and inPred:
                tp+=1
            elif not inGold and not inPred:
                tn+=1
            elif not inGold and inPred:
                fp+=1
            elif inGold and not inPred:
                fn+=1
    precision = tp / (tp+fp)
    recall = 0
    if (tp+fn)>0:
        recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
