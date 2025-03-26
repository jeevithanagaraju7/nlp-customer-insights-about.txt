# nlp_insights.py
from transformers import pipeline

# Fake customer comments (you could use real ones later)
comments = [
    "Love this payment app, so fast!",
    "Ugh, my transaction failed again",
    "Pretty good service, I guess"
]

# Set up the NLP tool (it guesses if comments are happy or sad)
nlp_tool = pipeline("sentiment-analysis")

# Check each comment
for comment in comments:
    result = nlp_tool(comment)[0]  # Get the first result
    feeling = result["label"]  # "POSITIVE" or "NEGATIVE"
    score = result["score"]    # How sure it is (0 to 1)
    print(f"Comment: {comment}")
    print(f"Feeling: {feeling}, Score: {score:.2f}\n")

# Save to a file (pretend it’s for a boss)
with open("customer_insights.txt", "w") as file:
    for comment in comments:
        result = nlp_tool(comment)[0]
        file.write(f"{comment} -> {result['label']}\n")
