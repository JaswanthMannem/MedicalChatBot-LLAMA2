from src.helper import load_pdf,text_split,hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.get("PINECONE_API_KEY")
os.environ.get("PINECONE_API_ENV")

extracted_data=load_pdf("C:\\Users\\DELL\\MedicalChatBot-LLAMA2\\data\\")
text_chunks=text_split(extracted_data)
embeddings=hugging_face_embedding()

index_name="medical-chatbot"
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embeddings,index_name=index_name)


