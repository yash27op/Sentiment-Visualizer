import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import pyttsx3

# Ensure nltk stopwords and VADER lexicon are downloaded
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Function to preprocess text (remove stopwords and punctuation)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Function to generate a word cloud image
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt)  # Display the WordCloud in Streamlit

# Function to visualize word frequency as a bar chart
def visualize_word_frequency(words):
    word_counts = Counter(words)

    # Bar Graph
    plt.figure(figsize=(10, 6))
    plt.bar(word_counts.keys(), word_counts.values(), color='skyblue')
    plt.title('Word Frequency - Bar Graph')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    st.pyplot(plt)  # Display the bar graph in Streamlit

# Function to visualize word frequency as a pie chart
def visualize_pie_chart(words):
    word_counts = Counter(words)
    plt.figure(figsize=(8, 8))
    plt.pie(word_counts.values(), labels=word_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Word Frequency - Pie Chart')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle
    st.pyplot(plt)  # Display the pie chart in Streamlit

# Function for sentiment analysis and detection of harmful words
def sentiment_analysis(text):
    # Analyze the sentiment of the text
    sentiment_scores = sia.polarity_scores(text)
    negative_words = []
    word_list = text.split()
    
    for word in word_list:
        # Check the sentiment intensity of each word
        word_score = sia.polarity_scores(word)
        if word_score['compound'] < -0.5:  # Adjust the threshold for harmful/negative words
            negative_words.append(word)

    return sentiment_scores, negative_words

# Function to speak harmful words using pyttsx3
def speak_harmful_words(negative_words):
    if negative_words:
        engine.say(f"The following harmful words were detected: {', '.join(negative_words)}")
        engine.runAndWait()

# Streamlit App Layout
st.title("Text Analysis and Sentiment Detection Dashboard")

# Input Section: Text Upload or Input
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method", ('Upload a File', 'Type/Paste Text'))

# Initialize text variable
text = None

if input_method == 'Upload a File':
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt'])
    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')  # Read the file content
else:
    text = st.sidebar.text_area("Input Text", "Enter or paste your text here")

# Ensure the text variable is defined before processing
if text:
    # Preprocess the text
    processed_words = preprocess_text(text)

    # Display Word Frequency
    st.subheader("Word Frequency Analysis")
    st.write(f"Total words processed: {len(processed_words)}")
    
    # Bar Chart Visualization
    st.subheader("Word Frequency - Bar Chart")
    visualize_word_frequency(processed_words)

    # Pie Chart Visualization
    st.subheader("Word Frequency - Pie Chart")
    visualize_pie_chart(processed_words)

    # Word Cloud Visualization
    st.subheader("Word Cloud")
    generate_wordcloud(' '.join(processed_words))

    # Sentiment Analysis
    st.subheader("Sentiment Analysis and Harmful Words Detection")
    sentiment_scores, negative_words = sentiment_analysis(text)

    # Display sentiment scores
    st.write("**Sentiment Scores:**")
    st.write(f"Positive: {sentiment_scores['pos']}")
    st.write(f"Neutral: {sentiment_scores['neu']}")
    st.write(f"Negative: {sentiment_scores['neg']}")
    st.write(f"Compound: {sentiment_scores['compound']}")

    # Display harmful/negative words and their count
    st.write(f"**Harmful Words Detected:** {len(negative_words)}")
    if negative_words:
        st.write(f"List of Harmful Words: {', '.join(negative_words)}")
        # Speak the harmful words
        st.write("Speaking harmful words detected...")
        speak_harmful_words(negative_words)

    # Raw Text Display
    st.subheader("Processed Text")
    st.write(' '.join(processed_words))

else:
    st.write("Please upload a file or enter text in the sidebar to begin.")
