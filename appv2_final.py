from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
#from streamlit_audiorecorder import st_audiorecorder as audiorecorder
from audio_recorder_streamlit import audio_recorder
import whisper
import tempfile
import os as os_module
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page config
st.set_page_config(
    page_title="  ROOTED",
    page_icon=" ",
    layout="centered"
)

st.markdown("""
<style>
    /* Main background - soft cream/off-white with subtle texture */
    .stApp {
        background: linear-gradient(135deg, #f5f1e8 0%, #e8e4d9 100%) !important;
    }
    
    /* Title styling - forest green */
    h1 {
        color: #2d5016 !important;
        font-family: 'Lato', serif !important;
        text-align: center !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0 !important;
        width: 100% !important;
        display: block !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #5a4a3a !important;
        font-style: italic;
        font-size: 1.1em;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Headers - earth tones */
    h2, h3, h4 {
        color: #5a4a3a !important;
        font-family: 'Lato', serif !important;
    }
    
    /* Force all text colors */
    .main p, .main div, .main span, .main li {
        color: #3d3d3d !important;
    }
    
    .main strong, .main b {
        color: #2d5016 !important;
        font-weight: 600 !important;
    }
    
    /* Markdown content */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #3d3d3d !important;
    }
    
    /* Input boxes - natural wood tone */
    .stTextArea textarea {
        background-color: #faf8f3 !important;
        border: 2px solid #8b7355 !important;
        border-radius: 10px !important;
        color: #3d3d3d !important;
        caret-color: #5a4a3a !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #a89885 !important;
        opacity: 0.7 !important;
    }

    /* Text area label - match the brown color */
    .stTextArea label, label {
        color: #3d3d3d !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - moss green */
    .stButton button {
        background: linear-gradient(135deg, #5a7c3e 0%, #4a6b2e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 12px 40px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #6a8c4e 0%, #5a7b3e 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.3) !important;
    }
    
    /* Sidebar - darker forest green */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3d5a2c 0%, #2d4a1c 100%) !important;
    }
    
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
        color: #f5f1e8 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #f5f1e8 !important;
        border-bottom: 2px solid #5a7c3e;
        padding-bottom: 10px;
    }
    
    /* Warning boxes - warm amber */
    .stAlert, [data-baseweb="notification"] {
        background-color: #fff4e6 !important;
        border-left: 4px solid #d4a574 !important;
        border-radius: 8px !important;
    }
    
    .stAlert *, [data-baseweb="notification"] * {
        color: #5a4a3a !important;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e8f5e9 !important;
        border-left: 4px solid #5a7c3e !important;
        border-radius: 8px !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #f1f8f4 !important;
        border-left: 4px solid #4a6b2e !important;
        border-radius: 8px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #faf8f3 !important;
        border: 1px solid #8b7355 !important;
        border-radius: 8px !important;
        color: #5a4a3a !important;
        font-weight: bold !important;
    }
    
    /* Audio recorder button styling */
    .stAudio {
        border: 2px solid #8b7355 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        background-color: #faf8f3 !important;
    }
    
    /* Add some padding to the main content */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Markdown link styling */
    a {
        color: #5a7c3e !important;
        text-decoration: none !important;
        font-weight: 500 !important;
    }
    
    a:hover {
        color: #4a6b2e !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("  ROOTED")
st.markdown('<p class="subtitle">🍃 Nature\'s remedies, digitally preserved 🍃</p>', unsafe_allow_html=True)

# Initialize the bot
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    needs_build = True
    
    if os.path.isdir(persist_dir):
        try:
            if any(os.listdir(persist_dir)):
                needs_build = False
        except Exception:
            needs_build = True

    if needs_build:
        with st.spinner("Building knowledge base from PDFs in ./data... This may take a few minutes on first run."):
            docs = []
            data_dir = "./data"
            
            # Debug: Check if directory exists
            if not os.path.isdir(data_dir):
                st.error(f"Data directory '{data_dir}' does not exist!")
            else:
                st.info(f"Found data directory: {data_dir}")
                
                # Debug: List all files
                all_files = []
                for root, _, files in os.walk(data_dir):
                    for f in files:
                        all_files.append(f)
                        if f.lower().endswith(".pdf"):
                            st.info(f"Loading PDF: {f}")
                            loader = PyPDFLoader(os.path.join(root, f))
                            docs.extend(loader.load())
                
                st.info(f"Files found in data directory: {all_files}")
                
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory=persist_dir)
                st.success(f"Knowledge base built from {len(docs)} PDF pages/entries.")
            else:
                st.warning("No PDFs found in ./data. The app will run, but results may be limited until documents are added.")
                vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    return vectorstore

vectorstore = load_vectorstore()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    #groq_api_key=os.environ.get("GROQ_API_KEY")
    groq_api_key=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
)

def get_remedy(user_issue):
    # Search for relevant documents
    docs = vectorstore.similarity_search(user_issue, k=8)
    
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are an expert herbalist and apothecary guide. Your job is to provide homemade herbal remedies based ONLY on the information in the provided books.

When a user describes a health issue, format your response EXACTLY like this:

**[Remedy Name]**

**Ingredients:**
List ALL ingredients with precise measurements. Be thorough and specific with ingredient names including ALL descriptive terms (essential oil, carrier oil, dried flowers, etc.):
- [quantity] [COMPLETE ingredient name]
- [quantity] [COMPLETE ingredient name]
(Include at least 3-5 ingredients when appropriate)

**Purchase Ingredients:**
For EACH ingredient above, create a shopping link using the FULL ingredient name (minus quantity):
- [complete ingredient name](https://www.google.com/search?tbm=shop&q=ingredient+name+with+plus+signs)

**Preparation:**
Provide DETAILED step-by-step instructions. Be thorough - include timing, temperatures, techniques:
1. [Detailed first step]
2. [Detailed second step]
3. [Continue with all necessary steps - at least 5-7 steps for most remedies]
4. [Include storage instructions]

**Application/Usage:**
How to use this remedy - dosage, frequency, best time to use, duration of treatment.

**Why it works:**
Explain the therapeutic properties of EACH ingredient and how they address the user's specific issue. Reference the traditional or scientific basis from the books.

**Safety warnings:**
Be thorough with contraindications:
- Who should avoid this remedy
- Potential side effects
- Drug interactions
- Pregnancy/nursing warnings
- Allergies to watch for

**Source:** [Book name, specific page numbers]

⚠️ Consult a healthcare provider before use.

IMPORTANT: Provide detailed, comprehensive information. Don't be brief. Include all relevant details from the books.

Context from herbalism books:
{context}

User's issue: {user_issue}

Now provide a DETAILED and COMPREHENSIVE remedy:"""
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content, docs

# Audio recorder
st.markdown('<h3 style="text-align: center;">Speak or Type Your Issue</h3>', unsafe_allow_html=True)

# Center the audio recorder to avoid a full-width bar
# col_left, col_center, col_right = st.columns([1, 2, 1])
# with col_center:
#     audio = audiorecorder("🎙️ Click to Record", "⏺️ Recording... Click to Stop")

# transcribed_text = ""

# if len(audio) > 0:
#     with col_center:
#         st.info(f"✅ Audio captured: {len(audio)} bytes")
        
#         # Play back the audio so user can verify
#         st.audio(audio.export().read(), format="audio/wav")
    
#     # Save audio to temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         audio.export(tmp_file.name, format="wav")
#         tmp_filename = tmp_file.name
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    audio_bytes = audio_recorder(text="Click to record", icon_size="2x")

transcribed_text = ""

if audio_bytes:
    with col_center:
        st.info(f"✅ Audio captured")
        
        # Play back the audio so user can verify
        st.audio(audio_bytes, format="audio/wav")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_filename = tmp_file.name
    
    # Transcribe using local Whisper
    with st.spinner("🎧 Transcribing your speech..."):
        model = whisper.load_model("base")
        result = model.transcribe(tmp_filename, language="en", fp16=False)
        transcribed_text = result["text"].strip()
    
    # Clean up temp file
    os_module.unlink(tmp_filename)
    
    if transcribed_text:
        st.success(f"✅ You said: **{transcribed_text}**")
    else:
        st.error("⚠️ No speech detected. Please try again.")

# Text input
user_issue = st.text_area(
    "Or type your health issue here:",
    value=transcribed_text,
    placeholder="Example: I have trouble sleeping and want a natural remedy",
    height=100
)

if st.button("Get Remedy", type="primary"):
    if user_issue:
        with st.spinner("🔍 Searching herbalism books..."):
            remedy, sources = get_remedy(user_issue)
            
            st.markdown("### Your Remedy:")
            st.markdown(remedy)
            
            # Always show disclaimer
            st.warning("⚠️ **IMPORTANT:** This is not medical advice. Please consult a healthcare provider for serious medical issues and before starting any herbal remedy, especially if you are pregnant, nursing, taking medications, or have existing health conditions.")
            
            # Show sources
            with st.expander("📚 View Sources from Books"):
                for i, doc in enumerate(sources[:3], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"- 📖 File: {doc.metadata.get('source_file', 'Unknown')}")
                    st.markdown(f"- 📄 Page: {doc.metadata.get('page', 'Unknown')}")
                    st.markdown(f"- 📝 Excerpt: _{doc.page_content[:200]}..._")
                    st.markdown("---")
    else:
        st.warning("⚠️ Please describe your health issue first.")

# Sidebar
with st.sidebar:
    st.markdown("### 🌿 About")
    st.markdown("""
    This application provides herbal remedies based on traditional herbalism books.
    
    **⚠️ Important Disclaimers:**
    - Not medical advice
    - Always consult healthcare provider
    - Check for allergies and drug interactions
    - Verify all information independently
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - I have trouble sleeping
    - What helps with headaches?
    - Natural remedy for digestive issues
    - How to make a cough syrup at home
    - Remedies for anxiety and stress
    """)
    
    st.markdown("---")
    
    st.markdown("### 🌱 Features")
    st.markdown("""
    - 🎤 Voice input support
    - 📚 RAG-powered book search
    - 🛍️ Direct shopping links
    - 🌿 Nature-inspired design
    """)
