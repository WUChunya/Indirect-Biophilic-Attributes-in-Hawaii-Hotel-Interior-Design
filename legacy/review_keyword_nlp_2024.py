
import pandas as pd
import re
from collections import Counter

# Function to search for attributes in reviews and classify them based on sentiment with word boundary checks
def analyze_reviews_optimized(reviews_df, attributes, positive_keywords, negative_keywords):
    results = {attribute: {'Positive Feedback Number': 0, 'Positive Character': Counter(), 
                           'Negative Feedback Number': 0, 'Negative Character': Counter()} 
               for attribute in attributes}

    # Compile regular expression patterns for efficient search
    positive_patterns = {word: re.compile(r'\b{}\b'.format(re.escape(word)), flags=re.IGNORECASE) for word in positive_keywords}
    negative_patterns = {word: re.compile(r'\b{}\b'.format(re.escape(word)), flags=re.IGNORECASE) for word in negative_keywords}

    for _, row in reviews_df.iterrows():
        # Convert feedback columns to string to handle NaNs
        positive_feedback = str(row['客户正面评价'])
        negative_feedback = str(row['客户负面评价'])
        overall_feedback = str(row['客户总评价'])

        for attribute in attributes:
            attribute_pattern = re.compile(r'\b{}\b'.format(re.escape(attribute)), flags=re.IGNORECASE)

            # Positive feedback analysis
            if attribute_pattern.search(positive_feedback):
                results[attribute]['Positive Feedback Number'] += 1
                for word, pattern in positive_patterns.items():
                    if pattern.search(positive_feedback):
                        results[attribute]['Positive Character'][word] += 1
            
            # Negative feedback analysis
            if attribute_pattern.search(negative_feedback):
                results[attribute]['Negative Feedback Number'] += 1
                for word, pattern in negative_patterns.items():
                    if pattern.search(negative_feedback):
                        results[attribute]['Negative Character'][word] += 1

            # Overall feedback analysis
            if attribute_pattern.search(overall_feedback):
                for word, pattern in positive_patterns.items():
                    if pattern.search(overall_feedback):
                        results[attribute]['Positive Feedback Number'] += 1
                        results[attribute]['Positive Character'][word] += 1
                for word, pattern in negative_patterns.items():
                    if pattern.search(overall_feedback):
                        results[attribute]['Negative Feedback Number'] += 1
                        results[attribute]['Negative Character'][word] += 1

    formatted_results = []
    for attr, data in results.items():
        if data['Positive Feedback Number'] > 0 or data['Negative Feedback Number'] > 0:
            formatted_results.append([
                attr, 
                data['Positive Feedback Number'], 
                ', '.join(f"{k}({v})" for k, v in data['Positive Character'].items()),
                data['Negative Feedback Number'], 
                ', '.join(f"{k}({v})" for k, v in data['Negative Character'].items())
            ])

    return formatted_results

# Function to process and analyze reviews for a list of hotels
def process_hotels(hotels, reviews_df, attributes, positive_keywords, negative_keywords):
    hotel_results = {}
    for hotel_name in hotels:
        # Filter reviews for the specific hotel
        hotel_reviews = reviews_df[reviews_df['酒店名称'] == hotel_name]
        # Analyze the reviews for the hotel
        hotel_results[hotel_name] = analyze_reviews_optimized(hotel_reviews, attributes, positive_keywords, negative_keywords)
    return hotel_results

# Function to sanitize sheet names by replacing invalid characters
def sanitize_sheet_name(name):
    invalid_chars = r'[]:*?/\\'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:31]  # Limit to 31 characters
