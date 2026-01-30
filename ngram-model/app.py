"""
Twi N-gram Language Model Demo
Streamlit application for demonstrating n-gram model capabilities
"""

import streamlit as st
import os
import math
import random
from collections import defaultdict

# ============== Model Functions ==============

def load_data(filepath):
    """Load text data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def split_to_sentences(data):
    """Split data by newline."""
    sentences = data.split('\n')
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def tokenize_sentences(sentences):
    """Tokenize sentences using whitespace splitting."""
    tokenized = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        if tokens:
            tokenized.append(tokens)
    return tokenized

def get_tokenized_data(data):
    """Split data into sentences and tokenize."""
    sentences = split_to_sentences(data)
    return tokenize_sentences(sentences)

def count_words(tokenized_sentences):
    """Count word frequencies."""
    word_counts = defaultdict(int)
    for sentence in tokenized_sentences:
        for token in sentence:
            word_counts[token] += 1
    return dict(word_counts)

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """Get words appearing at least count_threshold times."""
    word_counts = count_words(tokenized_sentences)
    return [word for word, cnt in word_counts.items() if cnt >= count_threshold]

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """Replace out-of-vocabulary words with <unk>."""
    vocabulary = set(vocabulary)
    replaced = []
    for sentence in tokenized_sentences:
        replaced_sentence = [token if token in vocabulary else unknown_token for token in sentence]
        replaced.append(replaced_sentence)
    return replaced

def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """Count all n-grams in the data."""
    n_grams = defaultdict(int)
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence) - n + 1):
            n_gram = sentence[i:i+n]
            n_grams[n_gram] += 1
    return dict(n_grams)

def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """Estimate probability of word given previous n-gram using k-smoothing."""
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    denominator = previous_n_gram_count + (k * vocabulary_size)
    
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
    numerator = n_plus1_gram_count + k
    
    return numerator / denominator

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0):
    """Estimate probabilities for all words in vocabulary."""
    previous_n_gram = tuple(previous_n_gram)
    vocab_extended = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocab_extended)
    
    probabilities = {}
    for word in vocab_extended:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability
    
    return probabilities

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token='<e>', k=1.0):
    """Calculate perplexity for a sentence."""
    n = len(list(n_gram_counts.keys())[0])
    sentence = [start_token] * n + sentence + [end_token]
    sentence = tuple(sentence)
    N = len(sentence)
    
    log_prob_sum = 0.0
    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        log_prob_sum += math.log(probability)
    
    return math.exp(-log_prob_sum / N)

def get_top_k_suggestions(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, top_k=5):
    """Get top-k word suggestions."""
    n = len(list(n_gram_counts.keys())[0])
    previous_tokens = ['<s>'] * n + previous_tokens
    previous_n_gram = previous_tokens[-n:]
    
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:top_k]

def generate_sentence(n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, max_length=20, temperature=1.0):
    """Generate a sentence by sampling from the model."""
    n = len(list(n_gram_counts.keys())[0])
    tokens = []
    
    for _ in range(max_length):
        previous_tokens = ['<s>'] * n + tokens
        previous_n_gram = previous_tokens[-n:]
        
        probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)
        
        # Apply temperature
        if temperature != 1.0:
            adjusted_probs = {w: p ** (1/temperature) for w, p in probabilities.items()}
            total = sum(adjusted_probs.values())
            probabilities = {w: p/total for w, p in adjusted_probs.items()}
        
        # Sample from distribution
        words = list(probabilities.keys())
        probs = list(probabilities.values())
        
        next_word = random.choices(words, weights=probs, k=1)[0]
        
        if next_word == '<e>':
            break
        if next_word != '<unk>':
            tokens.append(next_word)
    
    return ' '.join(tokens)

# ============== Load Model ==============

@st.cache_data
def load_model():
    """Load and prepare the n-gram model."""
    DATA_DIR = "data"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.twi")
    MIN_FREQ = 2
    
    # Load and tokenize
    train_raw = load_data(TRAIN_FILE)
    train_tokenized = get_tokenized_data(train_raw)
    
    # Build vocabulary
    vocabulary = get_words_with_nplus_frequency(train_tokenized, MIN_FREQ)
    train_data = replace_oov_words_by_unk(train_tokenized, vocabulary)
    
    # Count n-grams (using unigram as default - best perplexity)
    n = 1
    k = 0.01
    n_gram_counts = count_n_grams(train_data, n)
    n_plus1_gram_counts = count_n_grams(train_data, n + 1)
    vocab_size = len(vocabulary) + 2
    
    return {
        'n_gram_counts': n_gram_counts,
        'n_plus1_gram_counts': n_plus1_gram_counts,
        'vocabulary': vocabulary,
        'vocab_size': vocab_size,
        'k': k,
        'n': n
    }

# ============== Streamlit App ==============

def main():
    st.set_page_config(
        page_title="Twi N-gram Model Demo",
        page_icon="🇬🇭",
        layout="wide"
    )
    
    st.title("🇬🇭 Twi N-gram Language Model")
    st.markdown("*Interactive demonstration of an n-gram language model for Twi*")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    st.success(f"✅ Model loaded! Vocabulary: {len(model['vocabulary']):,} words")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Autocomplete", "✨ Generate Text", "📊 Score Sentence"])
    
    # Tab 1: Autocomplete
    with tab1:
        st.header("Word Autocomplete")
        st.markdown("Enter Twi text and see predictions for the next word.")
        
        user_input = st.text_input("Enter Twi text:", placeholder="e.g., na awurade")
        num_suggestions = st.slider("Number of suggestions:", 1, 10, 5)
        
        if user_input:
            tokens = user_input.lower().split()
            suggestions = get_top_k_suggestions(
                tokens, 
                model['n_gram_counts'], 
                model['n_plus1_gram_counts'],
                model['vocabulary'], 
                k=model['k'], 
                top_k=num_suggestions
            )
            
            st.subheader("Top Predictions:")
            for i, (word, prob) in enumerate(suggestions, 1):
                if word not in ['<e>', '<unk>']:
                    st.markdown(f"**{i}.** `{word}` — probability: {prob:.4f}")
    
    # Tab 2: Text Generation
    with tab2:
        st.header("Text Generation")
        st.markdown("Generate Twi sentences from the model.")
        
        col1, col2 = st.columns(2)
        with col1:
            max_len = st.slider("Maximum sentence length:", 5, 30, 15)
        with col2:
            temperature = st.slider("Temperature (creativity):", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🎲 Generate Sentence", type="primary"):
            with st.spinner("Generating..."):
                sentence = generate_sentence(
                    model['n_gram_counts'],
                    model['n_plus1_gram_counts'],
                    model['vocabulary'],
                    k=model['k'],
                    max_length=max_len,
                    temperature=temperature
                )
            st.success(f"**Generated:** {sentence}")
        
        st.markdown("---")
        st.markdown("*Tip: Higher temperature = more creative/random, lower = more predictable*")
    
    # Tab 3: Sentence Scoring
    with tab3:
        st.header("Sentence Perplexity Scoring")
        st.markdown("See how 'fluent' a sentence is according to the model. *Lower perplexity = better*")
        
        test_sentence = st.text_area("Enter a Twi sentence to score:", placeholder="e.g., na awurade ka kyerɛɛ")
        
        if test_sentence:
            tokens = test_sentence.lower().split()
            # Replace unknown words
            vocab_set = set(model['vocabulary'])
            tokens_processed = [t if t in vocab_set else '<unk>' for t in tokens]
            
            perplexity = calculate_perplexity(
                tokens_processed,
                model['n_gram_counts'],
                model['n_plus1_gram_counts'],
                model['vocab_size'],
                k=model['k']
            )
            
            st.metric("Perplexity", f"{perplexity:.2f}")
            
            if perplexity < 100:
                st.success("🟢 Very fluent!")
            elif perplexity < 500:
                st.info("🟡 Moderately fluent")
            else:
                st.warning("🔴 Less fluent / unusual")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built for NLP class presentation | Twi N-gram Language Model*")

if __name__ == "__main__":
    main()
