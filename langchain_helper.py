from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv


load_dotenv()

embeddings = HuggingFaceEmbeddings()

#video_url = "https://www.youtube.com/watch?v=dXxQ0LR-3Hg&t=1330s"

def create_vector_db(video_url:str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100 ) # Document 2 will have 100 charachters from Doc 1 as well

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db
def get_response_from_query(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    temperature = 0.7
    )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=""" 
        You are a helpful video assistant that can answer queries about videos
        based on the videos transcript.

        Answer the following question:{question}
        by searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you do not have enough information to answer the question
        just say " I Don't Know".



"""
    )

    chain = LLMChain(llm = llm, prompt = prompt)

    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace("\n", "")

    return response

