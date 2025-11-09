from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from audiorecorder import audiorecorder
import whisper
import tempfile
import os as os_module
import os

# Page config
st.set_page_config(
    page_title="Herbal Remedy Advisor",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Herbal Remedy Advisor")
st.markdown("*Get homemade herbal remedies based on traditional herbalism books*")

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
    temperature=0.4,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

def get_remedy(user_issue):
    # Search for relevant documents
    docs = vectorstore.similarity_search(user_issue, k=5)
    
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are an expert herbalist and apothecary guide. Your job is to provide homemade herbal remedies based ONLY on the information in the provided books.

When a user describes a health issue, provide a COMPREHENSIVE and DETAILED remedy following this format:

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
⚠️ Consult a healthcare provider before use.

**Source:** [Book name, specific page numbers]

IMPORTANT: Provide detailed, comprehensive information. Don't be brief. Include all relevant details from the books.

Context from herbalism books:
{context}

User's issue: {user_issue}

Now provide a DETAILED and COMPREHENSIVE remedy:"""
   
    # Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content, docs

# Audio recorder
st.markdown("### 🎤 Speak or Type Your Issue")

audio = audiorecorder("Click to Record", "Recording... Click to Stop")

transcribed_text = ""

if len(audio) > 0:
    st.info(f"Audio captured: {len(audio)} bytes")
    
    # Play back the audio so user can verify
    st.audio(audio.export().read(), format="audio/wav")
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio.export(tmp_file.name, format="wav")
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
            st.warning("⚠️ IMPORTANT: This is not medical advice. Please consult a healthcare provider for serious medical issues and before starting any herbal remedy, especially if you are pregnant, nursing, taking medications, or have existing health conditions.")
            
            # Show sources
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources[:3], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f"- File: {doc.metadata.get('source_file', 'Unknown')}")
                    st.markdown(f"- Page: {doc.metadata.get('page', 'Unknown')}")
                    st.markdown(f"- Excerpt: {doc.page_content[:200]}...")
                    st.markdown("---")
    else:
        st.warning("Please describe your health issue first.")

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This bot provides herbal remedies based on traditional herbalism books.
    
    **⚠️ Important:**
    - Not medical advice
    - Always consult healthcare provider
    - Check for allergies and drug interactions
    """)
    
    st.markdown("### Example Questions:")
    st.markdown("""
    - I have trouble sleeping
    - What helps with headaches?
    - Natural remedy for digestive issues
    - How to make a cough syrup at home
    """)
