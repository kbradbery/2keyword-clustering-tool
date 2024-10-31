```python
import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
import io
import base64
from datetime import datetime

st.set_page_config(page_title="Advanced SEO Analysis Suite", layout="wide")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("anthropic").get("ANTHROPIC_API_KEY"))

class SEOAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.clusters = []
        self.internal_links = []
        self.business_segments = []
        
    def analyze_intent(self, keyword: str) -> str:
        """Analyze search intent if not provided"""
        informational_markers = ['how', 'what', 'why', 'guide', 'tutorial']
        commercial_markers = ['best', 'top', 'vs', 'review', 'compare']
        transactional_markers = ['buy', 'price', 'cost', 'shop', 'purchase']
        
        keyword_lower = keyword.lower()
        
        if any(marker in keyword_lower for marker in informational_markers):
            return 'informational'
        elif any(marker in keyword_lower for marker in commercial_markers):
            return 'commercial'
        elif any(marker in keyword_lower for marker in transactional_markers):
            return 'transactional'
        else:
            return 'navigational'

    def create_mind_map(self, cluster):
        """Create interactive mind map for a cluster"""
        net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Add main topic
        net.add_node(cluster['name'], label=cluster['name'], color="#00ff1e")
        
        # Add keywords as sub-topics
        for kw in cluster['keywords']:
            net.add_node(kw['term'], 
                        label=f"{kw['term']}\n({kw['search_volume']} vol)", 
                        color="#00ff99")
            net.add_edge(cluster['name'], kw['term'])
        
        # Save to HTML file
        net.save_graph(f"mind_map_{cluster['name'].replace(' ', '_')}.html")
        
        return net.html

    def generate_internal_linking(self, clusters):
        """Generate internal linking suggestions"""
        linking_opportunities = []
        
        for cluster1 in clusters:
            for cluster2 in clusters:
                if cluster1 != cluster2:
                    relevance_score = self._calculate_relevance(cluster1, cluster2)
                    if relevance_score > 0.5:  # Threshold for relevance
                        linking_opportunities.append({
                            'source_cluster': cluster1['name'],
                            'target_cluster': cluster2['name'],
                            'relevance_score': relevance_score,
                            'link_type': self._determine_link_type(cluster1, cluster2),
                            'suggested_anchor': self._suggest_anchor(cluster1, cluster2)
                        })
        
        return linking_opportunities

    def _calculate_relevance(self, cluster1, cluster2):
        """Calculate relevance score between clusters"""
        # Implement relevance calculation logic
        common_terms = len(set(k['term'].lower().split()) & 
                         set(' '.join([k['term'].lower() for k in cluster2['keywords']]).split()))
        max_terms = max(len(' '.join([k['term'].lower() for k in cluster1['keywords']]).split()),
                       len(' '.join([k['term'].lower() for k in cluster2['keywords']]).split()))
        return common_terms / max_terms if max_terms > 0 else 0

    def _determine_link_type(self, cluster1, cluster2):
        """Determine the type of internal link"""
        if cluster1['intent'] == 'informational' and cluster2['intent'] == 'commercial':
            return 'Information to Product'
        elif cluster1['intent'] == 'commercial' and cluster2['intent'] == 'informational':
            return 'Product to Information'
        else:
            return 'Related Content'

    def _suggest_anchor(self, source_cluster, target_cluster):
        """Suggest anchor text for internal linking"""
        # Use primary keyword from target cluster as anchor
        primary_keyword = next((k['term'] for k in target_cluster['keywords'] 
                              if k.get('is_primary')), target_cluster['keywords'][0]['term'])
        return primary_keyword

    def export_to_excel(self, clusters, internal_links):
        """Export analysis to Excel with multiple sheets"""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Clusters Sheet
            clusters_data = []
            for cluster in clusters:
                for kw in cluster['keywords']:
                    clusters_data.append({
                        'Cluster': cluster['name'],
                        'Keyword': kw['term'],
                        'Search Volume': kw['search_volume'],
                        'Difficulty': kw['difficulty'],
                        'Intent': kw.get('intent', ''),
                        'Is Primary': kw.get('is_primary', False)
                    })
            
            pd.DataFrame(clusters_data).to_excel(writer, sheet_name='Keyword Clusters', index=False)
            
            # Content Suggestions Sheet
            content_data = []
            for cluster in clusters:
                content_data.append({
                    'Topic': cluster['name'],
                    'Content Type': cluster['content_suggestion']['type'],
                    'Structure': '\n'.join(cluster['content_suggestion']['structure']),
                    'Primary Keyword': next((k['term'] for k in cluster['keywords'] if k.get('is_primary')), ''),
                    'Search Intent': cluster['intent']
                })
            
            pd.DataFrame(content_data).to_excel(writer, sheet_name='Content Suggestions', index=False)
            
            # Internal Linking Sheet
            pd.DataFrame(internal_links).to_excel(writer, sheet_name='Internal Linking', index=False)
            
        return buffer

def main():
    st.title("🎯 Advanced SEO Analysis Suite")
    
    uploaded_file = st.file_uploader("Upload your keywords CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Validate and process data
        required_columns = ["keyword", "search_volume", "difficulty"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain: {', '.join(required_columns)}")
            return
            
        analyzer = SEOAnalyzer(df)
        
        # Add intent if not present
        if 'intent' not in df.columns:
            df['intent'] = df['keyword'].apply(analyzer.analyze_intent)
        
        # Analysis Options
        st.subheader("Analysis Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cluster_size = st.slider("Min Cluster Size", 2, 10, 3)
        with col2:
            similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.5)
        with col3:
            max_clusters = st.slider("Max Clusters", 5, 50, 20)
        
        if st.button("🔍 Analyze Keywords"):
            with st.spinner("Analyzing keywords..."):
                # Process in chunks and get clusters
                chunks = np.array_split(df, max(1, len(df) // 100))
                progress_bar = st.progress(0)
                
                all_results = []
                for i, chunk in enumerate(chunks):
                    try:
                        # Process chunk with Claude
                        prompt = f"""Analyze these keywords and create detailed clusters:
                        {chunk.to_string()}
                        
                        For each cluster:
                        1. Identify primary and secondary keywords
                        2. Determine search intent
                        3. Suggest content type and structure
                        4. Consider business relevance
                        
                        Return as JSON with schema:
                        {{
                            "clusters": [
                                {{
                                    "name": "topic",
                                    "keywords": [
                                        {{
                                            "term": "keyword",
                                            "search_volume": number,
                                            "difficulty": number,
                                            "intent": "intent",
                                            "is_primary": boolean
                                        }}
                                    ],
                                    "intent": "cluster_intent",
                                    "content_suggestion": {{
                                        "type": "content_type",
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
                        
                        chunk_results = json.loads(response.content)
                        all_results.extend(chunk_results['clusters'])
                        progress_bar.progress((i + 1) / len(chunks))
                    except Exception as e:
                        st.warning(f"Error processing chunk {i+1}: {str(e)}")
                        continue
                
                # Generate internal linking suggestions
                internal_links = analyzer.generate_internal_linking(all_results)
                
                # Display Results in Tabs
                tabs = st.tabs(["Clusters", "Mind Maps", "Internal Linking", "Visualizations"])
                
                with tabs[0]:
                    st.subheader("Keyword Clusters")
                    for cluster in all_results:
                        with st.expander(f"📑 {cluster['name']} ({cluster['intent']})"):
                            # Primary Keywords
                            st.write("**Primary Keywords:**")
                            primary_kw = [k for k in cluster['keywords'] if k.get('is_primary')]
                            st.dataframe(pd.DataFrame(primary_kw))
                            
                            # Secondary Keywords
                            st.write("**Secondary Keywords:**")
                            secondary_kw = [k for k in cluster['keywords'] if not k.get('is_primary')]
                            st.dataframe(pd.DataFrame(secondary_kw))
                            
                            # Content Suggestion
                            st.write("**Content Strategy:**")
                            st.write(f"Type: {cluster['content_suggestion']['type']}")
                            st.write("Structure:")
                            for section in cluster['content_suggestion']['structure']:
                                st.write(f"- {section}")
                
                with tabs[1]:
                    st.subheader("Topic Mind Maps")
                    for cluster in all_results:
                        with st.expander(f"🔄 {cluster['name']} Mind Map"):
                            mind_map_html = analyzer.create_mind_map(cluster)
                            components.html(mind_map_html, height=500)
                
                with tabs[2]:
                    st.subheader("Internal Linking Strategy")
                    st.dataframe(pd.DataFrame(internal_links))
                    
                with tabs[3]:
                    st.subheader("Cluster Visualizations")
                    
                    # Volume vs Difficulty Scatter Plot
                    fig = px.scatter(
                        pd.DataFrame([item for cluster in all_results for item in cluster['keywords']]),
                        x='search_volume',
                        y='difficulty',
                        color='intent',
                        hover_data=['term'],
                        title='Keyword Distribution: Volume vs Difficulty'
                    )
                    st.plotly_chart(fig)
                    
                    # Intent Distribution Pie Chart
                    intent_dist = pd.DataFrame([item for cluster in all_results for item in cluster['keywords']])['intent'].value_counts()
                    fig2 = px.pie(values=intent_dist.values, names=intent_dist.index, title='Search Intent Distribution')
                    st.plotly_chart(fig2)
                
                # Export Button
                buffer = analyzer.export_to_excel(all_results, internal_links)
                
                st.download_button(
                    label="📥 Download Complete Analysis",
                    data=buffer.getvalue(),
                    file_name=f"seo_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Add styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00ff99;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
```