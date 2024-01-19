import importlib
from typing import Tuple
from termcolor import colored
from colored import fg, attr as cAttr
from lexicalrichness import LexicalRichness
from flair.data import Sentence
from flair.models import SequenceTagger
from numpy import isclose
import nltk


def calc_compat_score(S: Tuple[str, float], bIsEmotionallyAvailable: bool, tc: float, lsm: float, cosim: float, dot: float, euclidean: float, convexHullJaccardRatio: float, mag1: float, mag2: float, w1: float, w2: float, w3: float, w4: float, w5: float, w6: float, w7: float, w8: float) -> float:
    color1 = fg("cyan")
    color2 = fg("red")
    color3 = fg("yellow")
    color4 = fg("green")
    color5 = fg("magenta")
    color6 = fg("blue")
    color7 = fg("white")
    reset = cAttr("reset")

    """
    compat = w_1 \cdot S + w_2 \cdot ratio_{hull} + w_3 \cdot euclidean_{norm} + w_4 \cdot (\cos(\text{sim}) \times \text{magRatio}) + w_5 \cdot dot_{norm}
    """

    if mag1 * mag2 > 0:
        dot = (dot / (mag1 * mag2) + 1) / 2
    else:
        dot = 0.5

    magRatio = min(mag1, mag2) / max(mag1, mag2) if max(mag1, mag2) != 0 else 0

    if S == 0:
        term1 = 0
    else:
        term1 = w1 * S[1]
    term2 = w2 * (1 if bIsEmotionallyAvailable else int(0))
    term3 = w3 * tc
    term4 = w4 * lsm
    term5 = w5 * convexHullJaccardRatio
    term6 = w6 * (1 / (1 + euclidean))
    term7 = w7 * cosim * magRatio
    term8 = w8 * dot

    if S == 0:
        print(f"{color1}attaction score: {S}{reset}")
    else:
        print(f"{color1}attaction score: {S[1]}{reset}")
    print(f"{color2}emotionally available: {bIsEmotionallyAvailable}{reset}")
    print(f"{color6}term count ratio: {tc}{reset}")
    print(f"{color3}lsm: {lsm}{reset}")
    print(f"{color4}convex hull jaccard ratio: {convexHullJaccardRatio}{reset}")
    print(f"{color5}euclidean: {euclidean}{reset}")
    print(f"{color6}cosim: {cosim}; magRatio: {magRatio}{reset}")
    print(f"{color7}normalized dot: {dot}{reset}")

    print("\n")

    print(f"{color1}term1: {term1}{reset}")
    print(f"{color2}term2: {term2}{reset}")
    print(f"{color3}term3: {term3}{reset}")
    print(f"{color4}term4: {term4}{reset}")
    print(f"{color5}term5: {term5}{reset}")
    print(f"{color6}term6: {term6}{reset}")
    print(f"{color7}term7: {term7}{reset}")
    print(f"{color2}term8: {term8}{reset}")

    scoresDict = {
        (0.9, 1.0): "veryCompatible",
        (0.85, 0.89): "compatible",
        (0.8, 0.84): "somewhat compatible",
        (0.4, 0.79): "incompatible",
        (0.0, 0.39): "fundamentally incompatible"
    }

    composite = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

    if composite < 0:
        composite = 0

    # for the general score

    def get_literal_score():
        for scoreRange, label in scoresDict.items():
            if scoreRange[0] <= composite <= scoreRange[1]:
                return label

    # score = w1 * S[1] + w2 * convexHullJaccardRatio + w3 * (1 / (1 + euclidean)) + w4 * cosim + w5 * dot
    return (composite, get_literal_score())

def calc_fwords(posTags: list):
    funcPosTags = {"CC", "DT", "EX", "IN", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WP$", "WRB"}

    wordCounts = {tag: 0 for tag in funcPosTags}
    for tags in posTags:
        for tag in tags:
            if tag in funcPosTags:
                wordCounts[tag] += 1
    return wordCounts

def calc_lsm(c1, c2):
    funcPosTags = {"CC", "DT", "EX", "IN", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WP$", "WRB"}
    c1 = {tag: c1[tag] / sum(c1.values()) for tag in c1 if tag in funcPosTags}
    c2 = {tag: c2[tag] / sum(c2.values()) for tag in c2 if tag in funcPosTags}
    lsm = 1 - sum([abs(c1[tag] - c2[tag]) for tag in funcPosTags]) / len(funcPosTags)
    return lsm

def calc_tc_ratio(tc1, tc2): # the term count ratio
    if tc1 == 0 or tc2 == 0:
        return 0
    return min(tc1, tc2) / max(tc1, tc2)

def calc_errr(msgs1, msgs2):
    raise NotImplementedError


if __name__ == "__main__":
    attr = importlib.import_module("attraction-classifier.infer").AttractionClassifier()
    userEmbedder = importlib.import_module("simtoon-embeddings.user_embedder").UserEmbedder()
    userUtils = importlib.import_module("simtoon-embeddings.user_utils").UserUtils()

    tagger = SequenceTagger.load("flair/pos-english")

    # csvPath = "/home/simtoon/git/ACARISv2/datasets/sarah/sarah.csv"
    # msgs, userID = userEmbedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="Username", msgCol="Content", sep=",")
    # users = [userID[0], userID[1]]

    # datPath = "/home/simtoon/git/ACARISv2/datasets/messages.dat"
    # users = ["simtoon1011#0", "simmiefairy#0"]
    # msgs, userID = userEmbedder.load_msgs_from_dat(datPath=datPath, limitToUsers=users)

    txtPath = "/home/simtoon/git/ACARISv2/datasets/allan/DMs.txt"
    users = ["Simtoon", "AllanMN"]
    msgs, userID = userEmbedder.load_direct_msgs_from_copied_discord_txt(txtPath=txtPath)

    print(f"Loaded {len(msgs[0] + msgs[1])} messages from {len(userID)} users")

    for user in users:
        if "simmie" in user:
            bIsEmotionallyAvailable = False
            break
        elif "sara" in user:
            bIsEmotionallyAvailable = False
            break
        else:
            bIsEmotionallyAvailable = False

    processedMsgs = [[], []]
    stemmer = nltk.SnowballStemmer("english")
    for msg in msgs[0]:
        msgWords = nltk.word_tokenize(str(msg))
        stemmedWords = [stemmer.stem(word) for word in msgWords]
        processedMsgs[0].append(" ".join(stemmedWords))
    for msg in msgs[1]:
        msgWords = nltk.word_tokenize(str(msg))
        stemmedWords = [stemmer.stem(word) for word in msgWords]
        processedMsgs[1].append(" ".join(stemmedWords))

    lex1, lex2 = LexicalRichness(" ".join(processedMsgs[0])), LexicalRichness(" ".join(processedMsgs[1]))

    t1, t2 = lex1.terms, lex2.terms
    tc = calc_tc_ratio(t1, t2)
    print(f"terms1: {t1}; terms2: {t2}; tc: {tc}")

    maas1, maas2 = lex1.Maas, lex2.Maas # ! the lower, the richer the vocab
    maasRatio = min(maas1, maas2) / max(maas1, maas2)
    print(f"maas1: {maas1}; maas2: {maas2}; maasRatio: {maasRatio}")

    posTags1, posTags2 = list(), list()
    for msg in processedMsgs[0]:
        sentence = Sentence(str(msg))
        tagger.predict(sentence)
        posTags1.append([sentence.labels[i].value for i in range(len(sentence.labels))])
    for msg in processedMsgs[1]:
        sentence = Sentence(str(msg))
        tagger.predict(sentence)
        posTags2.append([sentence.labels[i].value for i in range(len(sentence.labels))])

    c1, c2 = calc_fwords(posTags1), calc_fwords(posTags2)
    lsm = calc_lsm(c1, c2)
    print(f"lsm: {lsm}\nc1: {c1}\nc2: {c2}")

    emb1 = userEmbedder.gen_embs_from_observations(msgs[0], bStore=True, userID=userID[0])
    emb2 = userEmbedder.gen_embs_from_observations(msgs[1], bStore=True, userID=userID[1])

    meanEmb1Reduced, meanEmb2Reduced = userUtils.get_user_embs_mean(emb1, emb2, True, 2) # True for reduction and 2 for 2 UMAP dimensions

    comparison = userUtils.compare_two_users(emb1, emb2)

    cosim = comparison["cosim"]
    # TODO: dot is very often negative - investigate
    dot = comparison["dot"]
    euclidean = comparison["euclidean"]
    jaccards = comparison["jaccards"]
    convexHullJaccardRatio = sum(jaccards.values()) / len(jaccards)
    mag1 = comparison["magnitude1"]
    mag2 = comparison["magnitude2"]
    w1, w2, w3, w4, w5, w6, w7, w8 = 0.3, 0.18, 0.17, 0.15, 0.05, 0.05, 0.05, 0.05
    wSum = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8
    if not isclose(wSum, 1):
        raise ValueError(f"Weights must sum to 1\nThey now sum to: {wSum}") # sanity check
    S, _ = attr.classify_image("simone.png")
    for user in users:
        if "Allan" in user:
            S = 0 # not applicable
    if S != 0:
        S = (S[0]["label"], S[0]["score"] if S[0]["label"] == "pos" else 1 - S[0]["score"])


    print(colored(f"\nComposite score: {calc_compat_score(S=S, bIsEmotionallyAvailable=bIsEmotionallyAvailable, tc=tc, lsm=lsm, cosim=cosim, dot=dot, euclidean=euclidean, convexHullJaccardRatio=convexHullJaccardRatio, mag1=mag1, mag2=mag2, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6, w7=w7, w8=w8)}", "blue"))
