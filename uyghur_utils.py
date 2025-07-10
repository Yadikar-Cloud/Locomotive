import stanza
import langid
import torch

LANGUAGE_CODE = "ug"

with open("lexicon.dic", "r", encoding="utf-8") as f:
    wordlist = [line.strip() for line in f]

stanza.download(LANGUAGE_CODE)
use_gpu = torch.cuda.is_available()
nlp = stanza.Pipeline(LANGUAGE_CODE, use_gpu=use_gpu)

def is_uyghur_sentence(input_sentence, threshold=0.5):
    if langid.classify(input_sentence)[0] == LANGUAGE_CODE:
        doc = nlp(input_sentence)
        match_count = 0
        total_count = 0
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos != "PUNCT":
                    total_count += 1
                    if word.lemma in wordlist:
                        match_count += 1
        if total_count > 0:
            ratio = match_count / total_count
            return ratio >= threshold, ratio, match_count, total_count
        else:
            return False, 0.0, 0, 0
    else:
        return False, 0.0, 0, 0
