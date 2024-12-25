import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
import json
from typing import List, Dict
import networkx as nx
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Advanced SEO Clustering Tool", layout="wide")

# Initialize Anthropic API
anthropic = Anthropic(api_key=st.secrets.get("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") or st.secrets.get("anthropic").get("ANTHROPIC_API_KEY"))

class KeywordAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.clusters = []

    def analyze_chunk(self, chunk_df: pd.DataFrame) -> Dict:
        keywords_str = "\n".join([f"- {row['keyword']} (vol:{row['search_volume']}, diff:{row['difficulty']}, intent:{row['intent']})"
                                  for _, row in chunk_df.iterrows()])
        prompt = f"""Analyze these SEO keywords and create detailed clusters:
        {keywords_str}

        For each cluster:
        1. Identify primary and secondary keywords based on search volume and difficulty
        2. Group by user intent and topic relevance
        3. Suggest content structure and internal linking opportunities
        
        Return as JSON:
        {{
            "clusters": [
                {{
                    "name": "cluster topic",
                    "primary_keyword": {{
                        "term": "main keyword",
                        "volume": number,
                        "difficulty": number,
                        "intent": "intent type"
                    }},
                    "secondary_keywords": [
                        {{
                            "term": "keyword",
                            "volume": number,
                            "difficulty": number,
                            "intent": "intent type"
                        }}
                    ],
                    "content_suggestion": {{
                        "type": "content type",
                        "structure": ["section1", "section2"],
                        "internal_linking": [
                            {{
                                "from": "source keyword",
                                "to": "target keyword",
                                "relevance": "high/medium/low"
                            }}
                        ]
                    }}
                }}
            ]
        }}"""
        
        response = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.content)

    def create_mind_map(self, clusters: List[Dict]) -> go.Figure:
        G = nx.Graph()
        for cluster in clusters:
            G.add_node(cluster['name'], node_type='cluster')
            G.add_node(cluster['primary_keyword']['term'], node_type='primary')
            G.add_edge(cluster['name'], cluster['primary_keyword']['term'])
            for kw in cluster['secondary_keywords']:
                G.add_node(kw['term'], node_type='secondary')
                G.add_edge(cluster['primary_keyword']['term'], kw['term'])
        
        pos = nx.spring_layout(G)
        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_trace = go.Scatter(x=[], y=[], mode='markers+text', hoverinfo='text', textposition='bottom center',
                                 marker=dict(showscale=True, colorscale='Viridis', size=20))

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_colors = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_text.append(node)
            if G.nodes[node]['node_type'] == 'cluster':
                node_colors.append(0)
            elif G.nodes[node]['node_type'] == 'primary':
                node_colors.append(1)
            else:
                node_colors.append(2)

        node_trace.marker.color = node_colors
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(showlegend=False, hovermode='closest',
                                          margin=dict(b=0, l=0, r=0, t=0),
                                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        return fig

    def create_excel_report(self, clusters: List[Dict]) -> BytesIO:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create different sheets for clusters, keywords, content plans, and linking opportunities
            clusters_data = []
            for cluster in clusters:
                clusters_data.append({
                    'Cluster Name': cluster['name'],
                    'Primary Keyword': cluster['primary_keyword']['term'],
                    'Primary KW Volume': cluster['primary_keyword']['volume'],
                    'Primary KW Difficulty': cluster['primary_keyword']['difficulty'],
                    'Primary KW Intent': cluster['primary_keyword']['intent'],
                    'Secondary Keywords Count': len(cluster['secondary_keywords'])
                })
            pd.DataFrame(clusters_data).to_excel(writer, sheet_name='Clusters Overview', index=False)
            # Add more sheets for keyword details, content plans, and internal linking

        output.seek(0)
        return output

def main():
    st.title("🎯 Advanced SEO Keyword Clustering & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload keywords Excel file (max 10,000 keywords)", 
        type="xlsx",
        help="Required columns: keyword, search_volume, difficulty, intent"
    )
    
    if uploaded_file:
        try:
            # Read the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            required_columns = ["keyword", "search_volume", "difficulty", "intent"]
            
            if not all(col in df.columns for col in required_columns):
                st.error(f"Missing required columns. Please ensure your Excel has: {', '.join(required_columns)}")
                return
            
            if len(df) > 10000:
                st.error("Please limit your keywords to 10,000 or less.")
                return
            
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            analyzer = KeywordAnalyzer(df)
            
            if st.button("🔍 Analyze Keywords"):
                with st.spinner("Processing keywords..."):
                    chunks = np.array_split(df, max(1, len(df) // 500))
                    progress_bar = st.progress(0)
                    
                    all_results = []
                    for i, chunk in enumerate(chunks):
                        results = analyzer.analyze_chunk(chunk)
                        all_results.extend(results['clusters'])
                        progress_bar.progress((i + 1) / len(chunks))
                    
                    # Display analysis results in tabs
                    tabs = st.tabs(["Clusters", "Mind Map", "Internal Linking", "Content Strategy"])
                    
                    with tabs[0]:
                        st.subheader("Keyword Clusters")
                        for cluster in all_results:
                            with st.expander(f"📑 {cluster['name']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Primary Keyword:**")
                                    st.write(f"- Term: {cluster['primary_keyword']['term']}")
                                    st.write(f"- Volume: {cluster['primary_keyword']['volume']}")
                                    st.write(f"- Difficulty: {cluster['primary_keyword']['difficulty']}")
                                    st.write(f"- Intent: {cluster['primary_keyword']['intent']}")
                                
                                with col2:
                                    st.write("**Secondary Keywords:**")
                                    sec_kw_df = pd.DataFrame(cluster['secondary_keywords'])
                                    st.dataframe(sec_kw_df)
                    
                    with tabs[1]:
                        st.subheader("Topic Mind Map")
                        mind_map = analyzer.create_mind_map(all_results)
                        st.plotly_chart(mind_map, use_container_width=True)
                    
                    with tabs[2]:
                        st.subheader("Internal Linking Opportunities")
                        for cluster in all_results:
                            with st.expander(f"🔗 {cluster['name']} Links"):
                                linking_df = pd.DataFrame(cluster['content_suggestion']['internal_linking'])
                                st.dataframe(linking_df)
                    
                    with tabs[3]:
                        st.subheader("Content Strategy")
                        for cluster in all_results:
                            with st.expander(f"📝 {cluster['name']} Content Plan"):
                                st.write(f"**Content Type:** {cluster['content_suggestion']['type']}")
                                st.write("**Structure:**")
                                for section in cluster['content_suggestion']['structure']:
                                    st.markdown(f"- {section}")
                    
                    excel_file = analyzer.create_excel_report(all_results)
                    st.download_button(
                        label="📥 Download Complete Analysis",
                        data=excel_file,
                        file_name="seo_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
