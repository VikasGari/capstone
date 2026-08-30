import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import requests
from config.config_manager import ConfigManager

# Load global configuration
config_manager = ConfigManager()
api_cfg = config_manager.get_section("api")
host = api_cfg.get("host", "127.0.0.1")
port = int(api_cfg.get("port", 8000))
api_url = f"http://{host}:{port}"

st.set_page_config(
    page_title="Brokerage Policy & Trading Rules Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics (vibrant colors, clean cards, spacing)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A; /* Navy Blue */
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563; /* Gray */
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #3B82F6; /* Blue border */
        margin-bottom: 1rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .disclaimer-box {
        font-size: 0.8rem;
        color: #6B7280;
        background-color: #FEF3C7; /* Amber warning */
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 2rem;
        border-left: 4px solid #F59E0B;
    }
    .highlight {
        background-color: #FEF08A; /* Light Yellow highlight */
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Diagnostics and Control
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bullish.png", width=70)
    st.header("Control Center")
    st.info("Status: Decoupled FastAPI + Streamlit mode")
    
    st.markdown("---")
    st.subheader("System Status")
    
    # Check health of backend FastAPI API
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=2)
        if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
            st.success("FastAPI Backend: Online")
        else:
            st.warning("FastAPI Backend: Unhealthy status")
    except Exception:
        st.error("FastAPI Backend: Offline")
        st.caption(f"Could not connect to backend server at {api_url}")
        
    st.markdown("---")
    st.subheader("Ingestion Management")
    if st.button("Trigger Corpus Ingestion"):
        with st.spinner("Re-indexing policy corpus..."):
            try:
                ingest_resp = requests.post(f"{api_url}/ingest")
                if ingest_resp.status_code == 200:
                    data = ingest_resp.json()
                    st.success(f"Ingestion successful! Ingested {data.get('chunks_ingested')} chunks.")
                else:
                    st.error("Ingestion request failed.")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

# Main Layout
st.markdown('<div class="main-title">Brokerage Rules & Trading Policy Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Grounded Question-Answering for Support and Operations Staff with Clause-Level Citations</div>', unsafe_allow_html=True)

# Main Query Section
query_input = st.text_input(
    "Ask a question about trading policies, F&O margins, fees, account terms, or settlement timelines:",
    placeholder="e.g., What is the margin requirement for trading index futures?",
    key="user_query"
)

if query_input:
    with st.spinner("Analyzing rules and generating grounded answer..."):
        try:
            # Query FastAPI Backend
            payload = {"query": query_input}
            response = requests.post(f"{api_url}/query", json=payload)
            
            if response.status_code == 200:
                answer_data = response.json()
                
                # Setup 2 columns: Left for Answer, Right for Structured Metadata/Citations
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("Answer")
                    st.write(answer_data.get("answer"))
                    
                    # Highlighted Citations
                    citations = answer_data.get("citations", [])
                    if citations:
                        st.markdown("---")
                        st.subheader("Grounded Reference Snippets")
                        for idx, cit in enumerate(citations):
                            st.markdown(f"""
                            <div style="background-color: #F9FAFB; padding: 1rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #3B82F6;">
                                <span class="citation-badge">[{idx+1}] {cit['source']}</span>
                                <b>{cit['clause_id']}: {cit['clause_title']}</b>
                                <p style="font-style: italic; color: #374151; margin-top: 0.5rem; font-size: 0.95rem;">
                                    "... {cit['snippet']} ..."
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                with col_right:
                    # Grounding Metrics Card
                    is_suff = answer_data.get("is_sufficient", False)
                    conf = answer_data.get("grounding_confidence", 0.0)
                    
                    st.subheader("Grounding Confidence")
                    if is_suff:
                        st.metric(label="Confidence Rating", value=f"{conf*100:.1f}%")
                        st.success("Grounded in Corpus")
                    else:
                        st.metric(label="Confidence Rating", value="0.0%")
                        st.error("Abstained: Insufficient Context")
                        st.info("The query fell outside the scope of available exchange rules or margin policies.")

                    # Structured Fields
                    st.markdown("---")
                    st.subheader("Extracted Rule Entities")
                    
                    # Rules
                    rules = answer_data.get("applicable_rules", [])
                    if rules:
                        st.markdown("**Applicable Rules/Policies:**")
                        for r in rules:
                            st.write(f"- {r}")
                            
                    # Thresholds
                    thresholds = answer_data.get("thresholds_and_timelines", [])
                    if thresholds:
                        st.markdown("**Numerical Thresholds & Timelines:**")
                        for t in thresholds:
                            st.write(f"- :orange[{t}]")
                            
                    # Required Actions
                    actions = answer_data.get("required_actions", [])
                    if actions:
                        st.markdown("**Required Actions:**")
                        for a in actions:
                            st.write(f"- :red[{a}]")
                            
            elif response.status_code == 400:
                st.error("Invalid Request. Query cannot be empty.")
            else:
                st.error(f"Backend Server Error ({response.status_code}): {response.json().get('detail')}")
                
        except Exception as e:
            st.error("Communication Error: Could not reach the FastAPI Backend API.")
            st.caption(f"Details: {e}")
            st.info("Tip: Make sure the FastAPI backend is running via `python main.py --api` before querying.")

# Disclaimer bottom notice
st.markdown("""
<div class="disclaimer-box">
    <b>Legal Disclaimer:</b> This assistant retrieves rules, margins, and timeline procedures strictly from synthetic mock policies and exchange rulebooks. Generated contents are for educational and informational purposes only, and do not constitute financial, investment, transaction execution, or legal advice.
</div>
""", unsafe_allow_html=True)
