import json
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import praw
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

# Function to load credentials
def load_credentials(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        credentials = {}
        for line in lines:
            key, value = line.strip().split('=')
            credentials[key] = value
        return credentials

# Load Reddit API credentials from the credentials file
credentials = load_credentials("credentials_reddit.txt")
print("Credentials file loaded successfully.")

client_id = credentials["client_id"]
client_secret = credentials["client_secret"]

# Define user agent
user_agent = 'CollectionData'

# Initialize PRAW Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)
print("Reddit instance initialized successfully.")

# Variables for post criteria
max_word_count = 500
max_post_count = 10

# Search for posts related to a specific topic on Reddit
# query = 'Covid-19'
# query = 'War'
query = 'Canada'

posts = reddit.subreddit('all').search(query, limit=max_post_count)

# Convert collected data to JSON format
data_to_save = []
for post in posts:
    word_count = len(post.selftext.split())
    if word_count <= max_word_count and post.selftext:  
        post_data = {
            "text": post.selftext,
        }
        data_to_save.append(post_data)
        if len(data_to_save) >= max_post_count:
            break

# Print the total number of collected posts
print(f"Total number of collected posts: {len(data_to_save)}")

# Save data to a JSON file
filename = f"Document.json"
with open(filename, "w", encoding="utf-8") as file:
    json.dump(data_to_save, file, ensure_ascii=False, indent=4)

print("Data saved to a JSON file successfully.")

# Function to load Text Analytics API credentials
def load_text_analytics_credentials(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        credentials = {}
        for line in lines:
            key, value = line.strip().split('=')
            credentials[key] = value
        return credentials

# Load Text Analytics API credentials from the credential file
text_analytics_credentials = load_text_analytics_credentials("credentials.txt")
print("Text Analytics API credentials loaded successfully.")

# Initialize Text Analytics client
def initialize_text_analytics_client(credentials):
    key = credentials["key"]
    endpoint = credentials["endpoint"]
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Initialize Text Analytics client using loaded credentials
client = initialize_text_analytics_client(text_analytics_credentials)

# Read text from JSON file
file_path = "document.json"  # Assuming "document.json" is in the same directory as the script
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    documents = [item["text"] for item in data]

# Analyze sentiment
response = client.analyze_sentiment(
    documents=documents,
    language='en-US',
)

# Process response
sentiments = []
highest_sentiment_scores = []

for doc in response:
    print('Sentiment Analysis Outcome: {0}'.format(doc.sentiment)) 

    print('Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f}'.format(
        doc.confidence_scores.positive,
        doc.confidence_scores.neutral,
        doc.confidence_scores.negative
    ))
    print('-'*75)

    sentiments.append(doc.sentiment)

    sentences = doc.sentences
    sentence_sentiments = []  # Store sentiment with the highest score for each sentence
    for indx, sentence in enumerate(sentences):
        print('Sentence #{0}'.format(indx+1))
        print('Sentence Text: {0}'.format(sentence.text))
        print('Sentence scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f}'.format(
            sentence.confidence_scores.positive,
            sentence.confidence_scores.neutral,
            sentence.confidence_scores.negative
        ))
        
        # Determine the sentiment with the highest score for this sentence
        if sentence.confidence_scores.positive > sentence.confidence_scores.neutral and sentence.confidence_scores.positive > sentence.confidence_scores.negative:
            sentence_sentiment = 'positive'
        elif sentence.confidence_scores.neutral > sentence.confidence_scores.positive and sentence.confidence_scores.neutral > sentence.confidence_scores.negative:
            sentence_sentiment = 'neutral'
        else:
            sentence_sentiment = 'negative'
        
        sentence_sentiments.append(sentence_sentiment)
        
    highest_sentiment_scores.append(sentence_sentiments)
    print()

# Count the distribution of sentiments with the highest score
positive_count = sum([sentence_sentiments.count('positive') for sentence_sentiments in highest_sentiment_scores])
neutral_count = sum([sentence_sentiments.count('neutral') for sentence_sentiments in highest_sentiment_scores])
negative_count = sum([sentence_sentiments.count('negative') for sentence_sentiments in highest_sentiment_scores])

# Create word cloud
text = ' '.join(documents)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Save the plots to the same PDF file
with PdfPages('sentiment_analysis_output.pdf') as pdf:
    # Visualize distribution of sentiments with the highest score
    plt.figure(figsize=(8, 6))
    plt.bar(['Positive', 'Neutral', 'Negative'], [positive_count, neutral_count, negative_count], color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Distribution of Sentiments with Highest Score in Each Sentence')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    pdf.savefig()
    plt.close()

    # Visualize sentiment distribution
    plt.figure(figsize=(8, 6))
    plt.hist(sentiments, bins=3, color='skyblue', edgecolor='black', linewidth=1.2)
    plt.title('Sentiment Analysis Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(range(3), ['Negative', 'Neutral', 'Positive'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    pdf.savefig()
    plt.close()

    # Visualize word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    pdf.savefig()
    plt.close()


