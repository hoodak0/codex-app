from transformers import pipeline

LABEL_CATEGORIES = {
    "Geopolitical Risk": ["war", "conflict", "sanctions", "elections"],
    "Climate Signal": ["heatwave", "flood", "wildfire", "emissions"],
    "Civil Unrest": ["protest", "riot", "strike", "demonstration"],
    "Disinformation Watch": ["fake news", "propaganda", "troll", "narrative"]
}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_signals(headlines):
    results = []
    for headline in headlines:
        all_labels = list(LABEL_CATEGORIES.keys())

        # Get the actual title if headline is a dict
        if isinstance(headline, dict):
            headline_text = headline.get("title", "")
            url = headline.get("link", "#")
        else:
            headline_text = headline
            url = "https://www.google.com/search?q=" + "+".join(headline_text.split())

        result = classifier(headline_text, candidate_labels=all_labels, multi_label=False)
        label = result["labels"][0]
        score = round(result["scores"][0] * 5)

        results.append({
            "headline": headline_text,
            "label": label,
            "score": score,
            "url": url
        })

    return results
