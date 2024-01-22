import importlib
from typing import Tuple, Dict, List
from termcolor import colored
from colored import fg, attr as cAttr
from lexicalrichness import LexicalRichness
from flair.data import Sentence
from flair.models import SequenceTagger
from numpy import isclose
import nltk
from dotenv import load_dotenv
from torch import cat

load_dotenv()


def calc_compat_score(S: Tuple[str, float], bIsEmotionallyAvailable: bool, tc: float, lsm: float, errr: float, cosim: float, dot: float, euclidean: float, convexHullJaccardRatio: float, mag1: float, mag2: float, w1: float, w2: float, w3: float, w4: float, w5: float, w6: float, w7: float, w8: float, w9: float) -> float:
    """
    The function `calc_compat_score` calculates a compatibility score based on various input parameters
    and returns the score along with a label indicating the level of compatibility.
    
    @param S The parameter `S` is a tuple containing a string and a float. It represents the attraction
    score.
    @param bIsEmotionallyAvailable bIsEmotionallyAvailable is a boolean variable that indicates whether
    the person is emotionally available or not.
    @param tc term count ratio
    @param lsm The parameter "lsm" stands for "LSM" and represents a numerical value.
    @param errr The parameter "errr" represents the error rate reduction ratio.
    @param cosim The parameter `cosim` represents the cosine similarity between two vectors.
    @param dot The parameter "dot" represents the normalized dot product between two vectors.
    @param euclidean The parameter "euclidean" represents the Euclidean distance between two vectors.
    @param convexHullJaccardRatio The parameter "convexHullJaccardRatio" represents the ratio of the
    intersection of the convex hulls of two objects to the union of the convex hulls. It is used as a
    term in the compatibility score calculation.
    @param mag1 The parameter `mag1` represents the magnitude of the first vector.
    @param mag2 The parameter "mag2" represents the magnitude of the second vector.
    @param w1 The parameter w1 represents the weight assigned to the attraction score (S) in the
    compatibility calculation.
    @param w2 The parameter w2 represents the weight assigned to the term related to emotional
    availability in the compatibility score calculation.
    @param w3 The parameter w3 represents the weight assigned to the term count ratio (TCR) in the
    compatibility score calculation. It determines the importance of TCR in the overall score.
    @param w4 The parameter w4 represents the weight assigned to the LSM (linguistic style matching) score
    in the compatibility calculation. It determines the importance of the LSM score in the overall
    compatibility score.
    @param w5 The parameter w5 represents the weight assigned to the term related to the error rate
    reduction ratio (ERRR) in the compatibility score calculation.
    @param w6 The parameter w6 represents the weight assigned to the term involving the convex hull
    Jaccard ratio in the compatibility score calculation.
    @param w7 The parameter w7 represents the weight assigned to the term involving the euclidean
    distance in the compatibility score calculation.
    @param w8 The parameter w8 represents the weight assigned to the term involving the cosine
    similarity and magnitude ratio in the compatibility score calculation.
    @param w9 The parameter `w9` represents the weight assigned to the term `term9` in the compatibility
    score calculation.
    
    @return The function `calc_compat_score` returns a tuple containing the composite score and the
    label associated with that score.
    """
    color1 = fg("cyan")
    color2 = fg("red")
    color3 = fg("yellow")
    color4 = fg("green")
    color5 = fg("magenta")
    color6 = fg("blue")
    color7 = fg("white")
    reset = cAttr("reset")

    """
   compat = w_1 \cdot S + w_2 \cdot \epsilon + w_3 \cdot TCR + w_4 \cdot \text{LSM} + w_5 \cdot \text{ERRR} + w_6 \cdot ratio_{hull} + w_7 \cdot euclidean_{norm} + w_8 \cdot (\cos(\text{sim}) \times \text{magRatio}) + w_9 \cdot dot_{norm}
    """

    dot = (dot / (mag1 * mag2) + 1) / 2 if mag1 * mag2 > 0 else 0.5

    magRatio = min(mag1, mag2) / max(mag1, mag2) if max(mag1, mag2) != 0 else 0

    errr = 1 - errr

    term1 = 0 if S == 0 else w1 * S[1]
    term2 = w2 * (1 if bIsEmotionallyAvailable else int(0))
    term3 = w3 * tc
    term4 = w4 * lsm
    term5 = w5 * errr
    term6 = w6 * convexHullJaccardRatio
    term7 = w7 * (1 / (1 + euclidean))
    term8 = w8 * cosim * magRatio
    term9 = w9 * dot

    print(f"{color1}attaction score: {S if S == 0 else S[1]}{reset}")
    print(f"{color2}emotionally available: {bIsEmotionallyAvailable}{reset}")
    print(f"{color6}term count ratio: {tc}{reset}")
    print(f"{color3}lsm: {lsm}{reset}")
    print(f"{color2}errr: {errr}{reset}")
    print(f"{color4}convex hull jaccard ratio: {convexHullJaccardRatio}{reset}")
    print(f"{color5}euclidean: {euclidean}{reset}")
    print(f"{color6}cosim: {cosim}; magRatio: {magRatio}{reset}")
    print(f"{color7}normalized dot: {dot}{reset}")

    print("\n")

    print(f"{color1}term1: {term1}{reset}") # attraction score
    print(f"{color2}term2: {term2}{reset}") # emotional availability
    print(f"{color3}term3: {term3}{reset}") # term count ratio
    print(f"{color4}term4: {term4}{reset}") # linguistic style matching
    print(f"{color5}term5: {term5}{reset}") # estimated relevant response ratio
    print(f"{color6}term6: {term6}{reset}") # convex hull jaccard ratio
    print(f"{color7}term7: {term7}{reset}") # euclidean distance between the tips of the mean embedding vectors of the two users
    print(f"{color2}term8: {term8}{reset}") # cosine similarity between the mean embedding vectors of the two users
    print(f"{color3}term9: {term9}{reset}") # normalized dot product between the mean embedding vectors of the two users

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

    def get_literal_score() -> str:
        """
        The function `get_literal_score` returns the label associated with a given composite score
        range.

        @return a tuple containing the value of the variable "composite" and the result of calling the
        function "get_literal_score()".
        """
        for scoreRange, label in scoresDict.items():
            if scoreRange[0] <= composite <= scoreRange[1]:
                return label

    return (composite, get_literal_score())

def calc_fwords(posTags: List[List[str]]) -> Dict[str, int]:
    """
    The function `calc_fwords` takes a list of lists of part-of-speech tags as input and returns a
    dictionary that counts the occurrences of functional words in the input.

    @param posTags The `posTags` parameter is a list of lists, where each inner list represents the
    part-of-speech tags for a sentence. Each part-of-speech tag is represented as a string.

    @return The function `calc_fwords` returns a dictionary where the keys are part-of-speech tags
    (e.g., "CC", "DT", "EX") and the values are the counts of words that have those tags in the input
    `posTags`.
    """
    funcPosTags = {"CC", "DT", "EX", "IN", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WP$", "WRB"}

    wordCounts = {tag: 0 for tag in funcPosTags}
    for tags in posTags:
        for tag in tags:
            if tag in funcPosTags:
                wordCounts[tag] += 1
    return wordCounts

def calc_lsm(c1: Dict[str, float], c2: Dict[str, float]) -> float:
    """
    The function `calc_lsm` calculates the Language Style Matching (LSM) score between two dictionaries
    of functional part-of-speech tags.

    @param c1 The parameter `c1` is a dictionary where the keys are strings representing part-of-speech
    tags and the values are floats representing the counts of each tag in a text or corpus.
    @param c2 The parameter `c2` is a dictionary that represents the frequency of different parts of
    speech tags in a text. Each key in the dictionary is a part of speech tag, and the corresponding
    value is the frequency of that tag in the text.

    @return The function `calc_lsm` returns a float value, which represents the calculated LSM
    (Linguistic Similarity Measure) between two dictionaries `c1` and `c2`.
    """
    funcPosTags = {"CC", "DT", "EX", "IN", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "TO", "WDT", "WP", "WP$", "WRB"}
    c1 = {tag: c1[tag] / sum(c1.values()) for tag in c1 if tag in funcPosTags}
    c2 = {tag: c2[tag] / sum(c2.values()) for tag in c2 if tag in funcPosTags}
    lsm = 1 - sum([abs(c1[tag] - c2[tag]) for tag in funcPosTags]) / len(funcPosTags)
    return lsm

def calc_tc_ratio(tc1: int, tc2: int) -> float:
    """
    The function calculates the ratio of two given numbers, with a minimum value divided by the maximum
    value.

    @param tc1 The parameter `tc1` represents the first test case count, which is an integer value.
    @param tc2 The parameter `tc2` represents the total number of test cases that have passed.

    @return The function `calc_tc_ratio` returns a float value.
    """
    if tc1 == 0 or tc2 == 0:
        return 0
    return min(tc1, tc2) / max(tc1, tc2)

def map_msgs_to_clusters(msgs: List[str], labels: List[str]) -> Dict[str, str]:
    """
    The function `map_msgs_to_clusters` takes in two lists, `msgs` and `labels`, and returns a
    dictionary where each message in `msgs` is mapped to its corresponding label in `labels`.

    @param msgs A list of strings representing the messages or texts that need to be mapped to clusters
    or labels.
    @param labels The `labels` parameter is a list of strings that represents the labels or categories
    associated with each message in the `msgs` parameter.

    @return a dictionary where the keys are the messages (strings) and the values are the corresponding
    labels (strings).
    """
    return {msg: label for msg, label in zip(msgs, labels)}

def is_relevant_response(msg: Dict[str, str], response: Dict[str, str], cluster: str, responseCluster: str, threshold: int) -> bool:
    """
    The function `is_relevant_response` checks if a response is relevant based on the cluster, temporal
    difference, and threshold.

    @param msg A dictionary containing information about the original message. It has the following
    keys:
    @param response The `response` parameter is a dictionary that represents a response message. It
    contains key-value pairs where the keys are strings and the values are also strings. The dictionary
    contains information about the response, such as the timestamp of the response.
    @param cluster The "cluster" parameter represents the cluster or group to which the message belongs.
    It is a string value that helps categorize or group similar messages together.
    @param responseCluster The `responseCluster` parameter represents the cluster to which the response
    belongs. It is a string that identifies the cluster.
    @param threshold The threshold parameter is an integer that represents the maximum time difference,
    in seconds, between the timestamps of the message and the response for them to be considered
    relevant.

    @return a boolean value.
    """
    temporalDiff = response["timestamp"] - msg["timestamp"]
    return cluster == responseCluster and temporalDiff <= threshold

def calc_errr(msgs1: List[str], msgs2: List[str], labels1: List[str], labels2: List[str], timestamps1: List[int], timestamps2: List[int], threshold: int) -> float:
    """
    The `calc_errr` function calculates the Estimated Relevant Response Ratio (ERRR) between two sets of
    messages, labels, and timestamps, using a given threshold.

    @param msgs1 A list of messages from the first party. Each message is represented as a string.
    @param msgs2 The parameter `msgs2` is a list of strings representing the messages from the second
    party in a conversation.
    @param labels1 The parameter `labels1` is a list of labels corresponding to each message in `msgs1`.
    It is used to group similar messages together into clusters.
    @param labels2 The parameter `labels2` is a list of labels corresponding to the messages in `msgs2`.
    It is used to cluster the messages in `msgs2` into different groups or categories.
    @param timestamps1 The parameter `timestamps1` is a list of integers representing the timestamps of
    the messages in `msgs1`. Each timestamp corresponds to a message in `msgs1`.
    @param timestamps2 The parameter `timestamps2` is a list of integers representing the timestamps of
    the messages in `msgs2`.
    @param threshold The threshold parameter is an integer value that determines the maximum time
    difference (in seconds) allowed between a message and its corresponding response for them to be
    considered relevant. If the time difference between a message and its response exceeds the
    threshold, the response is considered irrelevant.

    @return The function `calc_errr` returns a float value, which represents the Estimated Relevant Response Ratio (ERRR).
    """

    # TODO: cover the case where the first party redirects due to a lack of interest perceived within the threshold, since *that* would likely result in a relevant response
    msgs1Clusters, msgs2Clusters = map_msgs_to_clusters(msgs1, labels1), map_msgs_to_clusters(msgs2, labels2)
    errrCount = 0

    msgs1 = [{"msg": msg, "timestamp": timestamp} for msg, timestamp in zip(msgs1, timestamps1)]
    msgs2 = [{"msg": msg, "timestamp": timestamp} for msg, timestamp in zip(msgs2, timestamps2)]

    for msg in msgs1:
        bIsRelevant = False
        for response in msgs2:
            if is_relevant_response(msg, response, msgs1Clusters[msg["msg"]], msgs2Clusters[response["msg"]], threshold):
                bIsRelevant = True
                break
        if not bIsRelevant:
            errrCount += 1

    return errrCount / len(msgs1) if msgs1 else 0 # the lower the ERRR, the better, because we're dividing the count of irrelevant responses by the total number of messages


if __name__ == "__main__":
    attr = importlib.import_module("attraction-classifier.infer").AttractionClassifier()
    userEmbedder = importlib.import_module("simtoon-embeddings.user_embedder").UserEmbedder()
    userUtils = importlib.import_module("simtoon-embeddings.user_utils").UserUtils()

    tagger = SequenceTagger.load("flair/pos-english")

    # csvPath = "/home/simtoon/git/ACARISv2/datasets/sarah/sarah.csv"
    # msgs, userID = userEmbedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="Username", msgCol="Content", sep=",")
    # users = [userID[0], userID[1]]

    datPath = "/home/simtoon/git/ACARISv2/datasets/messages.dat"
    users = ["simtoon1011#0", "simmiefairy#0"]
    msgs, userID, timestamps = userEmbedder.load_msgs_from_dat(datPath=datPath, limitToUsers=users)

    # txtPath = "/home/simtoon/git/ACARISv2/datasets/allan/DMs.txt"
    # users = ["Simtoon", "AllanMN"]
    # msgs, userID = userEmbedder.load_direct_msgs_from_copied_discord_txt(txtPath=txtPath)

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

    embsReduced, embs2Reduced = userUtils.get_user_embs(emb1, emb2, True, 2) # True for reduction and 2 for 2 UMAP dimensions

    hdb, labels = userUtils.cluster_embs(cat((embsReduced, embs2Reduced)), nClusters=100)
    user1Labels, user2Labels = labels[:len(embsReduced)], labels[len(embsReduced):]

    # the lower the ERRR, the better
    errr = calc_errr(msgs[0], msgs[1], user1Labels, user2Labels, timestamps[0], timestamps[1], 86400) # 86400 seconds in a day (make this a const)
    print(f"errr: {errr}")

    comparison = userUtils.compare_two_users(emb1, emb2)

    cosim = comparison["cosim"]
    # TODO: dot is very often negative - investigate
    dot = comparison["dot"]
    euclidean = comparison["euclidean"]
    jaccards = comparison["jaccards"]
    convexHullJaccardRatio = sum(jaccards.values()) / len(jaccards)
    mag1 = comparison["magnitude1"]
    mag2 = comparison["magnitude2"]
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = 0.2, 0.15, 0.19, 0.17, 0.14, 0.05, 0.05, 0.025, 0.025
    wSum = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
    if not isclose(wSum, 1):
        raise ValueError(f"Weights must sum to 1\nThey now sum to: {wSum}") # sanity check
    attraction = attr.classify_image("simone.png")
    if attraction is not None:
        S, _ = attraction
    else:
        S = 0 # should never happen
    for user in users:
        if "Allan" in user:
            S = 0 # not applicable
    if S != 0:
        S = (S[0]["label"], S[0]["score"] if S[0]["label"] == "pos" else 1 - S[0]["score"])


    print(colored(f"\nComposite score: {calc_compat_score(S=S, bIsEmotionallyAvailable=bIsEmotionallyAvailable, tc=tc, lsm=lsm, errr=errr, cosim=cosim, dot=dot, euclidean=euclidean, convexHullJaccardRatio=convexHullJaccardRatio, mag1=mag1, mag2=mag2, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, w6=w6, w7=w7, w8=w8, w9=w9)}", "blue"))
