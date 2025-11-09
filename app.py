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
    temperature=0.3,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

def get_remedy(user_issue):
    # Search for relevant documents
    docs = vectorstore.similarity_search(user_issue, k=5)
    
    # Build context from documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are an expert herbalist and apothecary guide. Your job is to provide homemade herbal remedies based ONLY on the information in the provided books.

	When a user describes a health issue, you should:
	1. Provide a specific homemade recipe/remedy from the books
	2. List exact ingredients and measurements (be specific: "lavender essential oil" not just "lavender", "coconut carrier oil" not just "coconut")
	3. RIGHT AFTER the ingredients list, add a "Purchase Ingredients" section with Google Shopping links for each ingredient.
   
   CRITICAL: Use the EXACT FULL ingredient name from your ingredients list in the search URL.
   
   Format:
   **Purchase Ingredients:**
   - [Full Ingredient Name Exactly As Listed Above](https://www.google.com/search?tbm=shop&q=full+ingredient+name+with+plus+signs)
   
   Examples:
   - If ingredient is "5 drops lavender essential oil" → link should be for "lavender essential oil"
   - If ingredient is "2 tbsp coconut carrier oil" → link should be for "coconut carrier oil"
   - If ingredient is "1 cup chamomile flowers" → link should be for "chamomile flowers"
   
	4. Then continue with step-by-step preparation instructions
	5. Explain why it works (based on the books)
	6. Include important safety warnings and contraindications
	7. Cite the source (book and page if available)
	8. Always end with "⚠️ Consult a healthcare provider before use."

Context from herbalism books:
{context}

User's issue: {user_issue}

Provide a detailed homemade remedy recipe with accurate shopping links:"""
    prompt = f"""You are an expert herbalist and apothecary guide. Your job is to provide homemade herbal remedies based ONLY on the information in the provided books.

When a user describes a health issue, you should:
1. Provide a specific homemade recipe/remedy from the books
2. List exact ingredients and measurements
3. Give clear step-by-step preparation instructions
4. Explain why it works (based on the books)
5. Include important safety warnings and contraindications
6. Cite the source (book and page if available)
7. Always end with "⚠️ Consult a healthcare provider before use."

Context from herbalism books:
{context}

User's issue: {user_issue}

Provide a detailed homemade remedy recipe:"""
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content, docs

# Audio recorder
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
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(tmp_filename, language="en", fp16=False)
        transcribed_text = result["text"].strip()
    
    # Clean up temp file
    import os as os_module
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
            
            # Extract and display shopping links for herbs mentioned
            st.markdown("---")
            st.markdown("#### 🛒 Purchase Ingredients")
            
Include only the actual herbs, roots, flowers, or natural ingredients - not equipment or containers.

Remedy:
{remedy}

Return ONLY the ingredient names, separated by commas. Example format: chamomile, valerian root, honey"""
            
            ingredient_response = llm.invoke(extraction_prompt)
            ingredients_text = ingredient_response.content.strip()
            
            # Parse ingredients
            ingredients = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
            
            if ingredients:
                st.markdown("Buy the ingredients for this remedy:")
                cols = st.columns(len(ingredients) if len(ingredients) <= 3 else 3)
                for i, ingredient in enumerate(ingredients):
                    with cols[i % 3]:
                        search_query = ingredient.replace(' ', '+')
                        google_shopping_link = f"https://www.google.com/search?tbm=shop&q={search_query}"
                        st.link_button(f"🛍️ {ingredient.title()}", google_shopping_link)
            else:
                st.info("No purchasable ingredients detected.")
            
            st.markdown("---")
            
            # Always show disclaimer
            st.warning("⚠️ IMPORTANT: This is not medical advice...")
            
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
