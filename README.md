#Hate Speech & Offensive Language Detection
A Hybrid Feature Engineering Approach
This repository features a robust machine learning pipeline designed to distinguish between Hate Speech, Offensive Language, and Neutral content. Unlike standard classifiers, this project leverages a "Hybrid Feature Set" that combines statistical, semantic, and linguistic properties of text.

✨ Key Features
Hybrid Vectorization: Combines TF-IDF (N-grams), Doc2Vec (Paragraph Embeddings), and Sentiment Scores (VADER).

Advanced Linguistic Engineering: Incorporates readability metrics like Flesch-Kincaid and syllable counts to identify structural patterns in toxic speech.

High-Speed Preprocessing: Utilizes a custom SpaCy pipeline for optimized tokenization and lemmatization.

Model Comparison: Evaluates performance across Logistic Regression, Random Forest, and Gaussian Naive Bayes.

🛠️ Technical Stack
NLP: SpaCy, NLTK (VADER), Textstat, Gensim (Doc2Vec).

ML: Scikit-learn (TF-IDF, Logistic Regression, Random Forest).

Visualization: Matplotlib, Seaborn, WordCloud.

📊 The Feature Engineering Pipeline
The project's strength lies in its Concatenated Feature Matrix. Each tweet is transformed into a high-dimensional vector containing:

N-gram TF-IDF: Captures specific keywords and phrases.

VADER Sentiment: Extracts Pos, Neg, Neu, and Compound scores.

Doc2Vec: Captures the semantic "meaning" and context of the tweet.

Meta-Features: Counts of @Mentions, #Hashtags, and URLs.

Readability Metrics: FKRA and FRE scores to detect complexity patterns.

🚀 Setup & Usage
1. Requirements
Bash

pip install pandas numpy spacy nltk gensim textstat scikit-learn matplotlib seaborn wordcloud
python -m spacy download en_core_web_sm
2. Execution
Run the script to process the data and train the final Logistic Regression classifier:

Bash

python hate_speech_detector.py
📈 Performance & Insights
The model includes a Normalized Confusion Matrix to track where the model struggles (e.g., distinguishing between "Hate" and "Offensive").

Insight: By combining Doc2Vec with Sentiment Analysis, the model significantly reduces False Positives in the "Neutral" category compared to using TF-IDF alone.
