import streamlit as st
from anthropic import Anthropic
import pandas as pd
import json

st.set_page_config(page_title="SEO Keyword Clustering Tool", layout="wide")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def analyze_keywords_with_claude(keywords_df):
    prompt = f"""Analyze these keywords and create clusters based on semantic similarity and search intent. 
    For each cluster, suggest content type and structure.
    Keywords with their metrics:
    {keywords_df.to_string()}
    
    Return the response as a JSON string with this structure:
    {{
        "clusters": [
            {{
                "name": "cluster name",
                "keywords": ["keyword1", "keyword2"],
                "intent": "search intent",
                "content_suggestion": {{
                    "type": "content type",
                    "structure": ["section1", "section2"]
                }}
            }}
        ]
    }}
    """
    
    response = anthropic.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content)

st.title("🎯 SEO Keyword Clustering Tool")

# File upload
uploaded_file = st.file_uploader("Upload your keywords CSV (columns: keyword, search_volume, difficulty, intent)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded keywords:")
    st.dataframe(df.head())
    
    if st.button("🔍 Analyze Keywords"):
        with st.spinner("Analyzing keywords with Claude..."):
            try:
                results = analyze_keywords_with_claude(df)
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["Clusters", "Content Suggestions"])
                
                with tab1:
                    for cluster in results["clusters"]:
                        with st.expander(f"📑 {cluster['name']} ({cluster['intent']})"):
                            st.write("Keywords:")
                            for kw in cluster["keywords"]:
                                st.markdown(f"- {kw}")
                
                with tab2:
                    for cluster in results["clusters"]:
                        with st.expander(f"📝 Content Plan: {cluster['name']}"):
                            st.write(f"**Content Type:** {cluster['content_suggestion']['type']}")
                            st.write("**Suggested Structure:**")
                            for section in cluster['content_suggestion']['structure']:
                                st.markdown(f"- {section}")
                
                # Add download button for results
                st.download_button(
                    "📥 Download Analysis",
                    json.dumps(results, indent=2),
                    "keyword_analysis.json",
                    "application/json"
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Simple instructions
with st.sidebar:
    st.markdown("""
    ### How to use:
    1. Prepare a CSV file with columns:
        - keyword
        - search_volume
        - difficulty
        - intent
    2. Upload the file
    3. Click 'Analyze Keywords'
    4. View clusters and content suggestions
    5. Download results
    
    ### Sample CSV format:
    ```
    keyword,search_volume,difficulty,intent
    seo tools,1200,45,informational
    best seo software,890,38,commercial
    ```
    """)