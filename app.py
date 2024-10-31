import streamlit as st
from anthropic import Anthropic
import pandas as pd
import json
from typing import List, Dict
import numpy as np

st.set_page_config(page_title="SEO Keyword Clustering Tool", layout="wide")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic").get("ANTHROPIC_API_KEY"))

def chunk_keywords(df: pd.DataFrame, chunk_size: int = 100) -> List[pd.DataFrame]:
    """Split keywords into smaller chunks to avoid token limits."""
    return np.array_split(df, max(1, len(df) // chunk_size))

def analyze_keywords_chunk(keywords_df: pd.DataFrame) -> Dict:
    """Analyze a single chunk of keywords."""
    # Convert to a more concise string format
    keywords_str = "\n".join([
        f"- {row['keyword']} (vol:{row['search_volume']}, diff:{row['difficulty']})"
        for _, row in keywords_df.iterrows()
    ])
    
    prompt = f"""Analyze these SEO keywords and create topical clusters. Keywords:
    {keywords_str}
    
    Create clusters based on semantic similarity and search intent.
    Return response as JSON with this structure:
    {{
        "clusters": [
            {{
                "name": "main topic",
                "keywords": ["keyword1", "keyword2"],
                "intent": "search intent",
                "content_suggestion": {{
                    "type": "content type",
                    "structure": ["section1", "section2"]
                }}
            }}
        ]
    }}
    
    Keep clusters focused and relevant. Prioritize user intent and search volume when grouping."""
    
    response = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content)

def merge_clusters(all_results: List[Dict]) -> Dict:
    """Merge clusters from multiple chunks intelligently."""
    merged_clusters = []
    seen_topics = set()
    
    for result in all_results:
        for cluster in result["clusters"]:
            # Check if similar topic exists
            similar_exists = False
            for existing in merged_clusters:
                if (existing["name"].lower() in cluster["name"].lower() or 
                    cluster["name"].lower() in existing["name"].lower()):
                    # Merge keywords
                    existing["keywords"] = list(set(existing["keywords"] + cluster["keywords"]))
                    similar_exists = True
                    break
            
            if not similar_exists and cluster["name"].lower() not in seen_topics:
                merged_clusters.append(cluster)
                seen_topics.add(cluster["name"].lower())
    
    return {"clusters": merged_clusters}

st.title("🎯 SEO Keyword Clustering Tool")

# File upload
uploaded_file = st.file_uploader("Upload your keywords CSV (columns: keyword, search_volume, difficulty, intent)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ["keyword", "search_volume", "difficulty"]
        
        # Validate columns
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
        else:
            # Show data preview
            st.write("Preview of uploaded keywords:")
            st.dataframe(df.head())
            
            total_keywords = len(df)
            st.info(f"Total keywords: {total_keywords}")
            
            if st.button("🔍 Analyze Keywords"):
                with st.spinner("Analyzing keywords with Claude..."):
                    # Process in chunks
                    chunks = chunk_keywords(df)
                    progress_bar = st.progress(0)
                    
                    all_results = []
                    for i, chunk in enumerate(chunks):
                        try:
                            chunk_result = analyze_keywords_chunk(chunk)
                            all_results.append(chunk_result)
                            progress_bar.progress((i + 1) / len(chunks))
                        except Exception as e:
                            st.warning(f"Warning: Error processing chunk {i+1}: {str(e)}")
                            continue
                    
                    # Merge results
                    if all_results:
                        final_results = merge_clusters(all_results)
                        
                        # Display results in tabs
                        tab1, tab2 = st.tabs(["Clusters", "Content Suggestions"])
                        
                        with tab1:
                            for cluster in final_results["clusters"]:
                                with st.expander(f"📑 {cluster['name']} ({cluster['intent']})"):
                                    st.write(f"**Keywords ({len(cluster['keywords'])}):**")
                                    keywords_df = pd.DataFrame(cluster["keywords"])
                                    st.dataframe(keywords_df)
                        
                        with tab2:
                            for cluster in final_results["clusters"]:
                                with st.expander(f"📝 Content Plan: {cluster['name']}"):
                                    st.write(f"**Content Type:** {cluster['content_suggestion']['type']}")
                                    st.write("**Suggested Structure:**")
                                    for section in cluster['content_suggestion']['structure']:
                                        st.markdown(f"- {section}")
                        
                        # Download results
                        st.download_button(
                            "📥 Download Analysis",
                            json.dumps(final_results, indent=2),
                            "keyword_analysis.json",
                            "application/json"
                        )
                    else:
                        st.error("No clusters could be created. Please check your data and try again.")
                        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Instructions sidebar
with st.sidebar:
    st.markdown("""
    ### 📝 Instructions
    
    1. Prepare your CSV file with columns:
        - keyword
        - search_volume
        - difficulty
        - intent (optional)
    
    2. Upload the file
    
    3. Click 'Analyze Keywords'
    
    ### 📊 Sample CSV Format:
    ```csv
    keyword,search_volume,difficulty,intent
    seo tools,1200,45,informational
    best seo software,890,38,commercial
    ```
    
    ### ℹ️ Tips
    - Larger keyword sets will take longer to process
    - Keywords are processed in chunks for better reliability
    - Similar topics are automatically merged
    """)