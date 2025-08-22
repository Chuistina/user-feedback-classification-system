import pandas as pd
import re
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"`.*?`", "", text)  # Remove code blocks (inline)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove code blocks (multiline)
    text = re.sub(r"[^a-z0-9\s\u4e00-\u9fa5]", "", text)  # Remove special characters, keep Chinese characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def clean_and_sample_data(input_file, output_dir="data/labeling", sample_size=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_file)
    
    # Drop duplicates based on title and body
    df.drop_duplicates(subset=["title", "body"], inplace=True)
    
    # Apply cleaning to title and body
    df["cleaned_title"] = df["title"].apply(clean_text)
    df["cleaned_body"] = df["body"].apply(clean_text)
    
    # Combine cleaned title and body for classification
    df["text_for_classification"] = df["cleaned_title"] + " " + df["cleaned_body"]
    
    # Sample data for labeling
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df.copy()
        
    output_file = os.path.join(output_dir, "label_task.csv")
    sample_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Cleaned and sampled {len(sample_df)} issues for labeling and saved to {output_file}")

if __name__ == "__main__":
    input_path = "data/processed/requests_requests_issues.csv"
    clean_and_sample_data(input_path)


