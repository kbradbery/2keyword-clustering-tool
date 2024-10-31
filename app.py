import streamlit as st
from anthropic import Anthropic
import pandas as pd
import json
from typing import List, Dict
import numpy as np

st.set_page_config(page_title="SEO Keyword Clustering Tool", layout="wide")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic").get("ANTHROPIC_API_KEY"))

def chunk_keywords(df: pd.DataFrame, chunk_size: int = 50) -> List[pd.DataFrame]:
    """Split keywords into smaller chunks to avoid token limits."""
    return np.array_split(df, max(1, len(df) // chunk_size))

def analyze_keywords_chunk(keywords_df: pd.DataFrame) -> Dict:
    """Analyze a single chunk of keywords."""
    # Convert to a more concise string format
    keywords_str = "\n".join([
        f"- {row['keyword']} (vol:{row['search_volume']}, diff:{row['difficulty']})"
        for _, row in keywords_df.iterrows()
    ])
    
    prompt = f"""Here are the SEO keywords to analyze:
    {keywords_str}
    
    Create topical clusters based on semantic similarity. For each cluster:
    1. Give it a descriptive name based on the main topic
    2. List all related keywords
    3. Identify the primary search intent
    4. Suggest a content type and structure

    Format your response as a valid JSON string with exactly this structure:
    {{
        "clusters": [
            {{
                "name": "name of cluster",
                "keywords": ["keyword1", "keyword2"],
                "intent": "search intent",
                "content_suggestion": {{
                    "type": "content type",
                    "structure": ["section1", "section2"]
                }}
            }}
        ]
    }}

    Make sure the JSON is valid and properly formatted. Include all keywords in appropriate clusters."""
    
    try:
        response = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON string from response
        response_text = response.content
        if isinstance(response_text, list):
            response_text = response_text[0].text
        
        # Find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in response")
            
    except Exception as e:
        st.error(f"Error parsing Claude's response: {str(e)}")
        return {"clusters": []}  # Return empty clusters on error

def merge_clusters(all_results: List[Dict]) -> Dict:
    """Merge clusters from multiple chunks intelligently."""
    merged_clusters = []
    seen_topics = set()
    
    for result in all_results:
        if not result or "clusters" not in result:
            continue
            
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
uploaded_file = st.file_uploader("Upload your keywords CSV (columns: keyword, search_volume, difficulty)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ["keyword", "search_volume", "difficulty"]
        
        # Validate columns
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
        else:
            # Clean and prepare data
            df = df.dropna(subset=["keyword"])
            df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0)
            df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce").fillna(0)
            
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
                            if chunk_result and "clusters" in chunk_result:
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
                                    cluster_df = pd.DataFrame(cluster["keywords"], columns=["keyword"])
                                    st.dataframe(cluster_df)
                        
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
    
    2. Upload the file
    
    3. Click 'Analyze Keywords'
    
    ### 📊 Sample CSV Format:
    ```csv
    keyword,search_volume,difficulty
    seo tools,1200,45
    best seo software,890,38
    ```
    
    ### ℹ️ Tips
    - Keep chunks small for better processing
    - Similar topics are automatically merged
    - Download results for backup
    """)