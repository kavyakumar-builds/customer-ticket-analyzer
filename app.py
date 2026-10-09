"""
Streamlit Web Application for AI-Powered Customer Support Ticket Analyzer.
Provides a clean, intuitive, and professional interface for analyzing support tickets.
"""

import json
import streamlit as st
from src.analyzer import TicketAnalyzer, AnalyzerError
from src.llm import LLMServiceError

# Page setup
st.set_page_config(
    page_title="AI-Powered Customer Support Ticket Analyzer",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for professional badge visuals and card layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .entity-tag {
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-weight: 500;
    }
    .priority-Critical { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; }
    .priority-High { background-color: #FFEDD5; color: #EA580C; border: 1px solid #FDBA74; }
    .priority-Medium { background-color: #FEF9C3; color: #CA8A04; border: 1px solid #FDE047; }
    .priority-Low { background-color: #DCFCE7; color: #16A34A; border: 1px solid #86EFAC; }
    .sentiment-Positive { background-color: #DCFCE7; color: #16A34A; }
    .sentiment-Neutral { background-color: #F1F5F9; color: #475569; }
    .sentiment-Negative { background-color: #FEE2E2; color: #DC2626; }
</style>
""", unsafe_allow_html=True)


# Preset samples for convenient viva demonstrations
SAMPLE_TICKETS = {
    "Select a pre-configured sample...": "",
    "1. Billing - Double Charge (High Priority)": (
        "I was charged twice for my subscription this month and I want a refund for the extra payment."
    ),
    "2. Technical - App Crash on Export (Critical Priority)": (
        "CRITICAL: The entire reporting dashboard crashes with error 500 whenever our finance team "
        "attempts to export monthly transaction data. This is completely blocking our month-end audit."
    ),
    "3. Account - Locked Out of 2FA (High Priority)": (
        "I recently lost my phone and cannot receive SMS verification codes to sign in. "
        "Please help me reset two-factor authentication so I can access my account."
    ),
    "4. Shipping - Package Tracking Not Updating (Medium Priority)": (
        "My order #88412 shipped 5 days ago via Express courier, but tracking number TRK-99210 "
        "still shows 'Label Created'. Can you please check with the logistics partner?"
    ),
    "5. Positive Feedback - Great Support Experience (Low Priority)": (
        "I just wanted to thank your agent Alex for resolving my configuration issue so quickly yesterday! "
        "The new interface is working smoothly."
    )
}


@st.cache_resource
def get_analyzer() -> TicketAnalyzer:
    """Cached initialization of the analyzer to avoid reloading files repeatedly."""
    return TicketAnalyzer()


def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Project Info")
        st.markdown(
            """
            **AI-Powered Customer Support Ticket Analyzer**  
            *Academic NLP Project for 7th Semester*

            ---
            **NLP Pipeline Flow:**
            1. Raw Ticket Input
            2. Strict Input Validation
            3. Prompt Templating (`prompts/`)
            4. Google Gemini API Inference
            5. JSON Response Parsing
            6. Pydantic Schema Validation
            7. Interactive Dashboard Display
            ---
            """
        )
        st.caption("Config: `config/config.yaml` | Engine: Google Gemini | Schema: `src/schemas.py`")

    # Header section
    st.markdown('<div class="main-header">🎫 AI-Powered Customer Support Ticket Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated NLP classification, entity extraction, sentiment analysis, '
        'and response drafting for support workflows.</div>',
        unsafe_allow_html=True
    )

    # Preset selection for fast viva testing
    selected_sample = st.selectbox(
        "💡 Quick Demo: Load a sample ticket or enter your own below:",
        options=list(SAMPLE_TICKETS.keys()),
        index=0
    )

    preset_text = SAMPLE_TICKETS[selected_sample]

    # Ticket input area
    ticket_input = st.text_area(
        label="Customer Support Ticket Content",
        value=preset_text,
        height=140,
        placeholder="Enter a customer support ticket here...",
        help="Paste raw customer inquiry, grievance, bug report, or billing request."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Ticket", type="primary", use_container_width=True)

    if analyze_clicked:
        # Validate that input is not empty
        if not ticket_input or not ticket_input.strip():
            st.warning("⚠️ Please enter a customer support ticket before clicking 'Analyze Ticket'.")
            return

        with st.spinner("Analyzing ticket with NLP engine and Google Gemini API..."):
            try:
                analyzer = get_analyzer()
                result = analyzer.analyze(ticket_input)
            except (AnalyzerError, LLMServiceError) as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
                return
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                return

        # Display results cleanly
        st.markdown("---")
        st.subheader("📊 Ticket Analysis Results")

        # Top Metric Cards: Category, Sentiment, Priority
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Classified Category", value=result.category.value)
        with m2:
            st.metric(label="Detected Sentiment", value=result.sentiment.value)
        with m3:
            st.metric(label="Assessed Priority", value=result.priority.value)

        st.markdown("<br>", unsafe_allow_html=True)

        # Extracted Entities
        st.markdown("##### 🏷️ Extracted Entities & Keywords")
        if result.entities:
            tags_html = "".join([f'<span class="entity-tag">{e}</span>' for e in result.entities])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("No specific domain entities or error codes detected in ticket text.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Issue Summary
        st.markdown("##### 📝 Issue Summary")
        st.info(result.summary)

        # Suggested Customer Response
        st.markdown("##### 💬 Suggested Customer Support Response")
        st.success(result.suggested_response)

        # Academic / JSON inspection expander
        with st.expander("🔍 View Validated Structured JSON (Schema Output)"):
            st.json(result.model_dump())


if __name__ == "__main__":
    main()
