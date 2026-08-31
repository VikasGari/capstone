import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
import requests
from config.config_manager import ConfigManager

# Load global configuration
config_manager = ConfigManager()
api_cfg = config_manager.get_section("api")
host = api_cfg.get("host", "127.0.0.1")
port = int(api_cfg.get("port", 8000))
api_title = api_cfg.get("title", "Brokerage Policy & Trading Rules Assistant")
api_url = f"http://{host}:{port}"

def answer_query(query):
    """Queries the FastAPI backend and formats the output into clean markdown."""
    if not query.strip():
        return "⚠️ Query cannot be empty. Please enter a question."
        
    try:
        payload = {"query": query}
        response = requests.post(f"{api_url}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            is_suff = data.get("is_sufficient", False)
            conf = data.get("grounding_confidence", 0.0)
            rules = data.get("applicable_rules", [])
            thresholds = data.get("thresholds_and_timelines", [])
            actions = data.get("required_actions", [])
            citations = data.get("citations", [])
            
            status_text = "✅ Grounded in Corpus" if is_suff else "❌ Abstained: Insufficient Context"
            
            # Format output markdown
            md_output = f"## 🤖 Assistant Answer\n{answer}\n\n"
            md_output += f"### 📊 Grounding Confidence: **{conf*100:.1f}%** ({status_text})\n\n"
            
            if rules or thresholds or actions:
                md_output += "### 📋 Extracted Policy Details\n"
                if rules:
                    md_output += "**Applicable Rules:**\n" + "\n".join([f"- {r}" for r in rules]) + "\n\n"
                if thresholds:
                    md_output += "**Numerical Thresholds & Timelines:**\n" + "\n".join([f"- {t}" for t in thresholds]) + "\n\n"
                if actions:
                    md_output += "**Required Actions:**\n" + "\n".join([f"- {a}" for a in actions]) + "\n\n"
            
            if citations:
                md_output += "### 📖 Reference Citations\n"
                for idx, cit in enumerate(citations):
                    md_output += f"**[{idx+1}] {cit['source']} — {cit['clause_id']}: {cit['clause_title']}**\n"
                    md_output += f"> *\"... {cit['snippet']} ...\"*\n\n"
                    
            return md_output
        elif response.status_code == 400:
            return "⚠️ Invalid Request. Query cannot be empty."
        else:
            return f"❌ Backend Error ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"❌ Communication Error: Could not connect to FastAPI backend at {api_url}.\nDetails: {e}"

def check_backend_status():
    """Checks the health of the FastAPI backend."""
    try:
        health_resp = requests.get(f"{api_url}/health", timeout=2)
        if health_resp.status_code == 200 and health_resp.json().get("status") == "healthy":
            return "🟢 FastAPI Backend: Online"
        return "🟡 FastAPI Backend: Unhealthy status"
    except Exception:
        return f"🔴 FastAPI Backend: Offline (Could not connect at {api_url})"

def trigger_ingestion():
    """Triggers clean corpus ingestion in the backend."""
    try:
        resp = requests.post(f"{api_url}/ingest")
        if resp.status_code == 200:
            data = resp.json()
            return f"✅ Ingestion successful! Ingested {data.get('chunks_ingested')} chunks."
        return f"❌ Ingestion failed with status code: {resp.status_code}"
    except Exception as e:
        return f"❌ Error connecting to backend: {e}"

# Build the Gradio blocks application
with gr.Blocks(title=api_title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 📈 {api_title}")
    gr.Markdown("A simple grounded question-answering assistant for trading policies, F&O margins, fees, and timelines.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Ask a question about trading policies:",
            placeholder="e.g., What is the margin requirement for trading index futures?",
            lines=2
        )
        
    submit_btn = gr.Button("Analyze Query", variant="primary")
    
    output_md = gr.Markdown(value="*Results will appear here after you ask a question.*")
    
    # Register events
    submit_btn.click(fn=answer_query, inputs=query_input, outputs=output_md)
    query_input.submit(fn=answer_query, inputs=query_input, outputs=output_md)
    
    with gr.Accordion("System Settings & Diagnostics", open=False):
        status_box = gr.Textbox(
            label="Backend Connection Status",
            value=check_backend_status(),
            interactive=False
        )
        refresh_btn = gr.Button("Refresh Status")
        refresh_btn.click(fn=check_backend_status, outputs=status_box)
        
        gr.Markdown("---")
        gr.Markdown("### Ingestion Control")
        ingest_btn = gr.Button("Trigger Corpus Re-Ingestion", variant="secondary")
        ingest_status = gr.Markdown()
        ingest_btn.click(fn=trigger_ingestion, outputs=ingest_status)

# Start the Gradio server
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8502)
