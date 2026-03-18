from strategy.portfolio_manager import Category

CRYPTO_KEYWORDS    = ["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto", "price"]
POLITICS_KEYWORDS  = ["president", "senate", "election", "nominee", "vance",
                      "harris", "newsom", "trump", "biden", "democrat", "republican",
                      "lebron", "brady", "kemp", "clinton", "obama"]
GEO_KEYWORDS       = ["iran", "russia", "ukraine", "ceasefire", "war", "nato",
                      "china", "taiwan", "israel", "hamas", "fed ", "federal reserve",
                      "netanyahu", "rate"]
ENTERTAIN_KEYWORDS = ["oscar", "grammy", "nba", "nfl", "champion", "winner",
                      "box office", "season", "barcelona", "chelsea", "newcastle"]


def classify_market(question: str) -> Category:
    q = question.lower()
    if any(k in q for k in CRYPTO_KEYWORDS):
        return Category.CRYPTO
    if any(k in q for k in GEO_KEYWORDS):
        return Category.GEOPOLITICS
    if any(k in q for k in POLITICS_KEYWORDS):
        return Category.POLITICS
    if any(k in q for k in ENTERTAIN_KEYWORDS):
        return Category.ENTERTAINMENT
    return Category.POLITICS
