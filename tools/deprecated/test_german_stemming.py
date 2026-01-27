from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("german")

words = [
    ("Arbeitsverträge", "Arbeitsvertrag"),
    ("Gewerbemietverträge", "Mietvertrag"),
    ("Schäden", "Schaden"),
    ("Beiträge", "Beitrag"),
    ("Marken", "Marke"),
    ("Patenten", "Patent"),
]

for w1, w2 in words:
    print(f"{w1} -> {stemmer.stem(w1)}")
    print(f"{w2} -> {stemmer.stem(w2)}")
    print(f"Match: {stemmer.stem(w1) == stemmer.stem(w2)}")
    print("-" * 20)
