import importlib
from typing import Tuple
from termcolor import colored
from colored import fg, attr as cAttr


def calc_compat_score(S: Tuple[str, float], cosim: float, dot: float, euclidean: float, convexHullJaccardRatio: float, mag1: float, mag2: float, w1: float, w2: float, w3: float, w4: float, w5: float) -> float:
    color1 = fg("cyan")
    color2 = fg("red")
    color3 = fg("yellow")
    color4 = fg("green")
    color5 = fg("magenta")
    reset = cAttr("reset")

    """
    compat = w_1 \cdot S + w_2 \cdot ratio_{hull} + w_3 \cdot euclidean_{norm} + w_4 \cdot (\cos(\text{sim}) \times \text{magRatio}) + w_5 \cdot dot_{norm}
    """

    if mag1 * mag2 > 0:
        dot = (dot / (mag1 * mag2) + 1) / 2
    else:
        dot = 0.5

    magRatio = min(mag1, mag2) / max(mag1, mag2) if max(mag1, mag2) != 0 else 0

    term1 = w1 * S[1]
    term2 = w2 * convexHullJaccardRatio
    term3 = w3 * (1 / (1 + euclidean))
    term4 = w4 * cosim * magRatio
    term5 = w5 * dot

    print(f"{color1}attaction score: {S[1]}{reset}")
    print(f"{color2}convex hull jaccard ratio: {convexHullJaccardRatio}{reset}")
    print(f"{color3}euclidean: {euclidean}{reset}")
    print(f"{color4}cosim: {cosim}; magRatio: {magRatio}{reset}")
    print(f"{color5}dot: {dot}{reset}")

    print("\n")

    print(f"{color1}term1: {term1}{reset}")
    print(f"{color2}term2: {term2}{reset}")
    print(f"{color3}term3: {term3}{reset}")
    print(f"{color4}term4: {term4}{reset}")
    print(f"{color5}term5: {term5}{reset}")

# TODO: the dict is not being interpreted correctly. it should be interpreted as ranges, not discrete values
    scoresDict = {
        "veryCompatible": 0.95,
        "compatible": 0.9,
        "somewhat compatible": 0.85,
        "incompatible": 0.3
    }

    composite = term1 + term2 + term3 + term4 + term5

    # for the general score

    def get_literal_score():
        for k, v in scoresDict.items():
            if composite >= v:
                return k
        return "incompatible"

    # score = w1 * S[1] + w2 * convexHullJaccardRatio + w3 * (1 / (1 + euclidean)) + w4 * cosim + w5 * dot
    return (composite, get_literal_score())


if __name__ == "__main__":
    attr = importlib.import_module("attraction-classifier.infer").AttractionClassifier()
    userEmbedder = importlib.import_module("simtoon-embeddings.user_embedder").UserEmbedder()
    userUtils = importlib.import_module("simtoon-embeddings.user_utils").UserUtils()

    csvPath = "/home/simtoon/git/ACARISv2/datasets/sarah/sarah.csv"
    msgs, userID = userEmbedder.load_msgs_from_csv(csvPath=csvPath, usernameCol="Username", msgCol="Content", sep=",")
    # print(f"Loaded {len(msgs[0] + msgs[1])} messages from {len(userID)} users")

    #datPath = "/home/simtoon/git/ACARISv2/datasets/messages.dat"
    #msgs, userID = userEmbedder.load_msgs_from_dat(datPath=datPath, limitToUsers=["simtoon1011#0", "simmiefairy#0"])

    emb1 = userEmbedder.gen_embs_from_observations(msgs[0], bStore=True, userID=userID[0])
    emb2 = userEmbedder.gen_embs_from_observations(msgs[1], bStore=True, userID=userID[1])

    meanEmb1Reduced, meanEmb2Reduced = userUtils.get_user_embs_mean(emb1, emb2, True, 2) # True for reduction and 2 for 2 UMAP dimensions

    comparison = userUtils.compare_two_users(emb1, emb2)

    cosim = comparison["cosim"]
    dot = comparison["dot"]
    euclidean = comparison["euclidean"]
    jaccards = comparison["jaccards"]
    convexHullJaccardRatio = sum(jaccards.values()) / len(jaccards)
    mag1 = comparison["magnitude1"]
    mag2 = comparison["magnitude2"]
    w1, w2, w3, w4, w5 = 0.5, 0.125, 0.125, 0.125, 0.125
    S, _ = attr.classify_image("sara.jpg")
    S = (S[0]["label"], S[0]["score"] if S[0]["label"] == "pos" else 1 - S[0]["score"])


    print(colored(f"\nComposite score: {calc_compat_score(S=S, cosim=cosim, dot=dot, euclidean=euclidean, convexHullJaccardRatio=convexHullJaccardRatio, mag1=mag1, mag2=mag2, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5)}", "blue"))
