from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("AI-POWERED CONTENT RECOMMENDATION SYSTEM")
print("TF-IDF + COSINE SIMILARITY ENGINE")
print("=" * 60)

items = [

    # Movies
    {"name": "Interstellar", "category": "Movie",
     "features": "sci-fi space future technology adventure"},

    {"name": "Inception", "category": "Movie",
     "features": "sci-fi dream mystery action thriller"},

    {"name": "The Matrix", "category": "Movie",
     "features": "sci-fi artificial intelligence technology action"},

    {"name": "Titanic", "category": "Movie",
     "features": "romance love emotional drama"},

    {"name": "The Notebook", "category": "Movie",
     "features": "romance relationship emotional love"},

    {"name": "John Wick", "category": "Movie",
     "features": "action crime thriller combat"},

    {"name": "The Dark Knight", "category": "Movie",
     "features": "action superhero crime thriller"},

    {"name": "The Conjuring", "category": "Movie",
     "features": "horror ghost supernatural scary"},

    {"name": "Insidious", "category": "Movie",
     "features": "horror suspense supernatural scary"},

    {"name": "3 Idiots", "category": "Movie",
     "features": "comedy friendship education drama"},

    # Songs
    {"name": "Shape of You", "category": "Song",
     "features": "pop dance romance music"},

    {"name": "Blinding Lights", "category": "Song",
     "features": "pop electronic energetic music"},

    {"name": "Perfect", "category": "Song",
     "features": "romance love emotional music"},

    {"name": "Love Yourself", "category": "Song",
     "features": "romance relationship pop music"},

    {"name": "Believer", "category": "Song",
     "features": "rock motivational energetic music"},

    {"name": "Thunder", "category": "Song",
     "features": "rock energetic powerful music"},

    {"name": "Lose Yourself", "category": "Song",
     "features": "hip-hop rap motivational music"},

    {"name": "Industry Baby", "category": "Song",
     "features": "hip-hop rap energetic music"},

    {"name": "Moonlight Sonata", "category": "Song",
     "features": "classical piano instrumental music"},

    {"name": "Canon in D", "category": "Song",
     "features": "classical orchestra instrumental music"},

    # Books
    {"name": "Atomic Habits", "category": "Book",
     "features": "self-help productivity success habits"},

    {"name": "Deep Work", "category": "Book",
     "features": "focus productivity self-help success"},

    {"name": "Rich Dad Poor Dad", "category": "Book",
     "features": "finance money investing wealth"},

    {"name": "Psychology of Money", "category": "Book",
     "features": "finance investing wealth money"},

    {"name": "Clean Code", "category": "Book",
     "features": "technology programming software coding"},

    {"name": "The Pragmatic Programmer", "category": "Book",
     "features": "software programming technology development"},

    {"name": "Harry Potter", "category": "Book",
     "features": "fantasy fiction magic adventure"},

    {"name": "The Hobbit", "category": "Book",
     "features": "fantasy fiction adventure journey"},

    {"name": "Steve Jobs", "category": "Book",
     "features": "biography entrepreneur technology innovation"},

    {"name": "Wings of Fire", "category": "Book",
     "features": "biography inspiration leadership success"},

    # Games
    {"name": "GTA V", "category": "Game",
     "features": "action open-world crime adventure"},

    {"name": "Valorant", "category": "Game",
     "features": "action shooting strategy multiplayer"},

    {"name": "Call of Duty", "category": "Game",
     "features": "action combat shooting warfare"},

    {"name": "Minecraft", "category": "Game",
     "features": "creativity exploration adventure building"},

    {"name": "Red Dead Redemption 2", "category": "Game",
     "features": "adventure story open-world action"},

    {"name": "Need for Speed", "category": "Game",
     "features": "racing cars speed competition"},

    {"name": "Forza Horizon 5", "category": "Game",
     "features": "racing driving open-world cars"},

    {"name": "FIFA 24", "category": "Game",
     "features": "sports football competition multiplayer"},

    {"name": "NBA 2K24", "category": "Game",
     "features": "sports basketball competition multiplayer"},

    {"name": "Civilization VI", "category": "Game",
     "features": "strategy planning empire management"}
]

trending_items = [
    "Interstellar",
    "Atomic Habits",
    "Harry Potter",
    "Shape of You",
    "GTA V"
]

while True:

    print("\nINGESTION STAGE")
    print("Enter at least 3 interests")

    interest1 = input("Interest 1: ").lower()
    interest2 = input("Interest 2: ").lower()
    interest3 = input("Interest 3: ").lower()

    user_profile = interest1 + " " + interest2 + " " + interest3

    print("\nPIPELINE EXECUTION")
    print("1. Ingestion Completed")
    print("2. TF-IDF Feature Extraction")
    print("3. Vector Mapping")
    print("4. Cosine Similarity Scoring")

    documents = [user_profile]

    for item in items:
        documents.append(item["features"])

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    user_vector = tfidf_matrix[0]
    item_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(
        user_vector,
        item_vectors
    )[0]

    recommendations = []

    for i in range(len(items)):
        recommendations.append(
            (
                items[i]["name"],
                items[i]["category"],
                similarity_scores[i]
            )
        )

    print("5. Sorting Recommendations")

    recommendations.sort(
        key=lambda x: x[2],
        reverse=True
    )

    highest_score = recommendations[0][2]

    print("6. Top-N Filtering")

    print("\n" + "=" * 60)
    print("RECOMMENDATION REPORT")
    print("=" * 60)

    if highest_score < 0.05:

        print("\nCOLD START DETECTED")
        print("No strong preference match found.")

        print("\nTrending Recommendations:")
        for item in trending_items:
            print("-", item)

    else:

        print("\nTop 10 Recommendations:\n")

        for name, category, score in recommendations[:10]:

            print(
                f"{name} ({category}) "
                f"| Similarity Score = {score:.2f}"
            )

    print("\nSYSTEM FEATURES USED")
    print("- Content-Based Filtering")
    print("- TF-IDF Weighting")
    print("- Vector Mapping")
    print("- Cosine Similarity")
    print("- Ranking Pipeline")
    print("- Cold Start Handling")

    choice = input("\nRun Again? (yes/no): ").lower()

    if choice != "yes":
        print("\nThank you for using the Recommendation System.")
        break