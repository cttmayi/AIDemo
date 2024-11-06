# Paraphrase Mining

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single list of sentences - Possible tens of thousands of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

paraphrases = paraphrase_mining(model, sentences)

for paraphrase in paraphrases:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))