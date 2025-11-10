from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
#from audiorecorder import audiorecorder
from audio_recorder_streamlit import audio_recorder as audiorecorder
import whisper
import tempfile
import os as os_module
import os

# Page config
st.set_page_config(
    page_title="  ROOTED",
    page_icon=" ",
    layout="centered"
)

# Custom CSS for nature theme with incense animation
st.markdown("""
<style>
    /* Main background - soft cream/off-white with subtle texture */
    .stApp {
        background: linear-gradient(135deg, #f5f1e8 0%, #e8e4d9 100%);
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
        color: #5a4a3a;
        font-style: italic;
        font-size: 1.1em;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Headers - earth tones */
    h2, h3 {
        color: #5a4a3a !important;
        font-family: 'Lato', serif !important;
    }
        /* All remedy output text - dark brown for readability */
    .main p {
        color: #3d3d3d !important;
    }
    
    .main li {
        color: #3d3d3d !important;
    }
    
    .main strong {
        color: #2d5016 !important;
        font-weight: 600 !important;
    }
    
    .main h1, .main h2, .main h3, .main h4 {
        color: #5a4a3a !important;
        display: block !important;
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
    .stTextArea label {
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
    
    [data-testid="stSidebar"] * {
        color: #f5f1e8 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #f5f1e8 !important;
        border-bottom: 2px solid #5a7c3e;
        padding-bottom: 10px;
    }
    
    /* Warning boxes - warm amber */
    .stAlert {
        background-color: #fff4e6 !important;
        border-left: 4px solid #d4a574 !important;
        border-radius: 8px !important;
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
    
    # /* Incense smoke animation */
    # @keyframes smoke {
    #     0% {
    #         transform: translateY(0) translateX(0) scale(1);
    #         opacity: 0.7;
    #     }
    #     25% {
    #         transform: translateY(-80px) translateX(15px) scale(1.3);
    #         opacity: 0.5;
    #     }
    #     50% {
    #         transform: translateY(-160px) translateX(-10px) scale(1.6);
    #         opacity: 0.3;
    #     }
    #     75% {
    #         transform: translateY(-240px) translateX(20px) scale(1.9);
    #         opacity: 0.1;
    #     }
    #     100% {
    #         transform: translateY(-320px) translateX(-5px) scale(2.2);
    #         opacity: 0;
    #     }
    # }
    
    # .smoke {
    #     position: fixed;
    #     bottom: 70px;
    #     left: calc(50% - 2px);
    #     width: 15px;
    #     height: 15px;
    #     background: radial-gradient(circle, rgba(180,180,180,0.6) 0%, rgba(150,150,150,0.3) 50%, transparent 100%);
    #     border-radius: 50%;
    #     animation: smoke 10s infinite ease-in-out;
    #     pointer-events: none;
    #     z-index: 9998;
    #     filter: blur(3px);
    # }
    
    # .smoke:nth-child(2) {
    #     animation-delay: 3.3s;
    #     left: calc(50% - 6px);
    # }
    
    # .smoke:nth-child(3) {
    #     animation-delay: 6.6s;
    #     left: calc(50% + 2px);
    # }
    
    # /* Incense stick */
    # .incense-stick {
    #     position: fixed;
    #     bottom: 20px;
    #     left: 50%;
    #     transform: translateX(-50%);
    #     width: 3px;
    #     height: 50px;
    #     background: linear-gradient(to bottom, #654321 0%, #4a2f1a 100%);
    #     border-radius: 2px;
    #     z-index: 9999;
    #     box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    # }
    
    # .incense-tip {
    #     position: absolute;
    #     top: -4px;
    #     left: 50%;
    #     transform: translateX(-50%);
    #     width: 7px;
    #     height: 7px;
    #     background: radial-gradient(circle, #ff6600 0%, #ff3300 60%, #cc0000 100%);
    #     border-radius: 50%;
    #     box-shadow: 
    #         0 0 8px #ff6600, 
    #         0 0 15px #ff3300,
    #         0 0 20px rgba(255, 51, 0, 0.5);
    #     animation: glow 2s infinite ease-in-out;
    # }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 
                0 0 8px #ff6600, 
                0 0 15px #ff3300,
                0 0 20px rgba(255, 51, 0, 0.5);
            transform: translateX(-50%) scale(1);
        }
        50% {
            box-shadow: 
                0 0 15px #ff6600, 
                0 0 25px #ff3300, 
                0 0 35px rgba(255, 0, 0, 0.7);
            transform: translateX(-50%) scale(1.1);
        }
    }
    
    /* Add some padding to the main content to avoid incense overlap */
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

<!-- Incense stick and smoke animation -->
<div class="incense-stick">
    <div class="incense-tip"></div>
</div>
<div class="smoke"></div>
<div class="smoke"></div>
<div class="smoke"></div>
""", unsafe_allow_html=True)

st.title("  ROOTED")
st.markdown('<p class="subtitle">🍃 Nature\'s remedies, digitally preserved 🍃</p>', unsafe_allow_html=True)

# Initialize the bot
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    groq_api_key=os.environ.get("GROQ_API_KEY")
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
st.markdown("### Speak or Type Your Issue")

audio = audiorecorder("🎙️ Click to Record", "⏺️ Recording... Click to Stop")

transcribed_text = ""

if audio is not None:
    st.info(f"✅ Audio captured: {len(audio)} bytes")
    
    # Play back the audio so user can verify
    #st.audio(audio.export().read(), format="audio/wav")
    st.audio(audio, format="audio/wav")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        #audio.export(tmp_file.name, format="wav")
        tmp_file.write(audio)
        tmp_file.flush()
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
