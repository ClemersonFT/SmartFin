import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuração pro groq trabaia

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Ó, as funções de transcreve ta aqui

def transcrever(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3"
        )
    return transcription.text

def resumir(texto):

    prompt = f"""
    Resuma esta reunião.

    Destaque:
    - decisões
    - tarefas
    - pontos importantes

    Texto:
    {texto}
    """

    resposta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return resposta.choices[0].message.content


def salvar_rag(texto):
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(texto)
    db = Chroma.from_texts(
        chunks,
        embedding,
        persist_directory="chroma_db"
    )
    db.persist()


def perguntar_rag(pergunta):
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding
    )
    docs = db.similarity_search(pergunta, k=3)
    contexto = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Responda usando o contexto abaixo.

    Contexto:
    {contexto}

    Pergunta:
    {pergunta}
    """

    resposta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return resposta.choices[0].message.content


#Daqui pra baixo é pra edita o frontzin q não ta la muito trabalhado na beleza

st.title("FinSmart")
st.subheader("IA para documentação de reuniões")

audio = st.file_uploader(
    "Envie uma reunião",
    type=["mp3", "wav", "m4a"]
)

if audio:
    os.makedirs("uploads", exist_ok=True)
    caminho = f"uploads/{audio.name}"
    with open(caminho, "wb") as f:
        f.write(audio.read())

    if st.button("Processar reunião"):
        with st.spinner("Transcrevendo..."):
            texto = transcrever(caminho)
        st.success("Transcrição concluída!")
        st.subheader("Transcrição")
        st.write(texto)

        with st.spinner("Gerando resumo..."):
            resumo = resumir(texto)
        st.subheader("Resumo")
        st.write(resumo)
        salvar_rag(texto)
        st.success("Reunião salva no RAG!")


#Ó ana aqui edita o chat

st.divider()
st.subheader("Consultar reuniões")
pergunta = st.text_input("Faça uma pergunta")
if st.button("Perguntar"):
    resposta = perguntar_rag(pergunta)
    st.write(resposta)