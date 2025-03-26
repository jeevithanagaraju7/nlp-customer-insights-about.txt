# nlp_insights.py
from transformers import pipeline

# Fake customer comments (replace with real ones later if needed)
comments = [
    "Love this payment app, so fast!",
    "Ugh, my transaction failed again",
    "Pretty good service, I guess"
]

# Set up the NLP tool to guess if comments are happy or sad
nlp_tool = pipeline("sentiment-analysis")

# Check each comment and print results
for comment in comments:
    result = nlp_tool(comment)[0]  # Get the first result
    feeling = result["label"]      # "POSITIVE" or "NEGATIVE"
    score = result["score"]        # Confidence score (0 to 1)
    print(f"Comment: {comment}")
    print(f"Feeling: {feeling}, Score: {score:.2f}\n")

# Save results to a file for a boss to see
with open("customer_insights.txt", "w") as file:
    for comment in comments:
        result = nlp_tool(comment)[0]
        file.write(f"{comment} -> {result['label']}\n")
