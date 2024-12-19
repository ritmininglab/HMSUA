from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from nltk.translate import meteor
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
from rouge import Rouge
import numpy as np
from datasets import load_metric


meteor2 = load_metric("meteor")


def calculate_bleu(candidate, references):
    if len(references) == 0 or len(references[0]) == 0 or len(candidate) == 0:
        return [1, 1, 1, 1, 1]
    reference = [word_tokenize(ref) for ref in references]
    candi = word_tokenize(candidate)
    score1 = sentence_bleu(reference, candi, weights=[1, 0, 0, 0])
    score2 = sentence_bleu(reference, candi, weights=[0, 1, 0, 0])
    score3 = sentence_bleu(reference, candi, weights=[0, 0, 1, 0])
    score4 = sentence_bleu(reference, candi, weights=[0, 0, 0, 1])
    score = np.exp(np.log([score1, score2, score3]).mean())
    return [score, score1, score2, score3, score4]


def calculate_meteor(candidate, references):
    if len(references) == 0 or len(references[0]) == 0 or len(candidate) == 0:
        return 1
    candis = [candidate.strip().split()]
    output = []
    for ref in references:
        words = ref.strip().split()
        result = meteor2.compute(predictions=candis, references=[[words]])
        output.append(result["meteor"])
    return np.max(output)


def calculate_cider(candidate, references):
    if len(references) == 0 or len(references[0]) == 0 or len(candidate) == 0:
        return 1
    candidates = preprocess_mono_sents([candidate])
    mult_references = preprocess_mult_sents([references])
    _, output = cider_d(candidates, mult_references)
    return output.numpy()[0]


def calculate_rouge(candidate, references):
    """
    candidate, reference: generated and ground-truth sentences
    """
    rouge = Rouge()
    score_l = 0
    if len(references) == 0 or len(references[0]) == 0 or len(candidate) == 0:
        return 1
    for ref in references:
        scores = rouge.get_scores(candidate, ref)
        score_l = max(score_l, scores[0]["rouge-l"]["f"])
    return score_l
