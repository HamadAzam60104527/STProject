import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from collections import Counter, defaultdict
import random
import re

# Set page config
st.set_page_config(page_title="YouTube Transcript Sentiment Analyzer", layout="wide")
st.title("YouTube Transcript Sentiment Analysis")

# --- Utility: Extract video ID from YouTube URL ---
def extract_youtube_id(url):
    """
    Extracts the YouTube video ID from a URL.
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Input: YouTube Video URL
youtube_url = st.text_input("Enter YouTube Video URL:", value="https://www.youtube.com/watch?v=SCwN0_ZXwec")
video_id = extract_youtube_id(youtube_url)

if video_id:
    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])

        # Chunk text
        def chunk_text(text, max_words=10):
            words = text.split()
            return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

        chunks = chunk_text(text)

        # Load sentiment pipeline
        st.info("Loading sentiment model...")
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

        label_map = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

        sentiment_counts = Counter()
        detailed_results = []
        grouped_by_label = defaultdict(list)

        with st.spinner("Analyzing sentiment..."):
            for i, chunk in enumerate(chunks):
                result = sentiment_pipeline(chunk)[0]
                label = label_map[result['label']]
                score = result['score']

                sentiment_counts[label] += 1
                detailed_results.append((i + 1, label, score, chunk))
                grouped_by_label[label].append((score, chunk))

        # Show all chunks
        st.subheader("Chunk-by-Chunk Sentiment")
        for idx, label, score, chunk in detailed_results:
            st.markdown(f"**Chunk {idx}** — *{label}* (score: {score:.3f})")
            st.write(f"{chunk}")

        # Sampled results
        st.subheader("Sampled Sentiment Examples (Random)")
        for category in ["Positive", "Negative", "Neutral"]:
            st.markdown(f"**{category} Samples:**")
            samples = random.sample(grouped_by_label[category], min(5, len(grouped_by_label[category])))
            for idx, (score, text) in enumerate(samples, start=1):
                st.write(f"{idx}. Score: {score:.3f}")
                st.write(f"`{text}`")

        # Overall Analysis
        st.subheader("Overall Sentiment Analysis")
        total_classified = sentiment_counts["Positive"] + sentiment_counts["Negative"]

        st.write(f"Positive: {sentiment_counts['Positive']} chunks")
        st.write(f"Negative: {sentiment_counts['Negative']} chunks")
        st.write(f"Neutral: {sentiment_counts['Neutral']} chunks (excluded from overall)")

        if sentiment_counts["Positive"] > sentiment_counts["Negative"]:
            overall_sentiment = "Positive"
        elif sentiment_counts["Negative"] > sentiment_counts["Positive"]:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral (equal positive and negative)"

        st.success(f"Final Overall Sentiment (excluding neutral): **{overall_sentiment}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.warning("Please enter a valid YouTube video URL.")
