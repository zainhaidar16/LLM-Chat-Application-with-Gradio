import os
import torch
import gradio as gr
from typing import List, Optional

# LangChain and ML imports
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import (
    HuggingFaceEmbeddings, 
    OpenAIEmbeddings
)
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline, OpenAI
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)

class DocumentChatApp:
    def __init__(
        self, 
        documents_dir: str = './data', 
        persist_dir: str = './chroma_db',
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        search_kwargs: dict = {"k": 3}
    ):
        """
        Initialize the Document Chat Application with customizable parameters
        
        Args:
            documents_dir (str): Directory containing input documents
            persist_dir (str): Directory to persist vector database
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between text chunks
            search_kwargs (dict): Configuration for document retrieval
        """
        self.DOCUMENTS_DIRECTORY = documents_dir
        self.PERSIST_DIRECTORY = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_kwargs = search_kwargs
        
        # Supported models
        self.EMBEDDING_MODELS = {
            "HuggingFace MiniLM": "all-MiniLM-L6-v2",
            "HuggingFace Paraphrase": "paraphrase-MiniLM-L3-v2",
            "OpenAI Embeddings": "text-embedding-ada-002"
        }
        
        self.LLM_MODELS = {
            "DialoGPT Small": "microsoft/DialoGPT-small",
            "DialoGPT Medium": "microsoft/DialoGPT-medium",
            "Pythia 70M": "EleutherAI/pythia-70m",
            "OpenAI GPT-3.5": "gpt-3.5-turbo"
        }
        
        # Initialize key components
        self.vectorstore = None
        self.chat_chain = None
        
    def load_and_process_documents(self) -> List:
        """
        Load and process documents from the specified directory
        
        Returns:
            List of processed document chunks
        """
        # Create directory if it doesn't exist
        os.makedirs(self.DOCUMENTS_DIRECTORY, exist_ok=True)
        
        # Load documents
        loader = DirectoryLoader(
            self.DOCUMENTS_DIRECTORY, 
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    
    def create_vector_store(
        self, 
        texts: List, 
        embedding_model: str = "HuggingFace MiniLM"
    ):
        """
        Create vector store from processed documents
        
        Args:
            texts (List): Processed document chunks
            embedding_model (str): Selected embedding model
        """
        # Select embedding model
        if embedding_model == "OpenAI Embeddings":
            # Requires OPENAI_API_KEY
            embeddings = OpenAIEmbeddings()
        else:
            # HuggingFace embeddings
            model_name = self.EMBEDDING_MODELS.get(embedding_model, "all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        
        # Create and persist vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings, 
            persist_directory=self.PERSIST_DIRECTORY
        )
        self.vectorstore.persist()
    
    def initialize_llm(
        self, 
        llm_model: str = "DialoGPT Small", 
        temperature: float = 0.7
    ):
        """
        Initialize Language Model
        
        Args:
            llm_model (str): Selected language model
            temperature (float): Creativity/randomness of responses
        
        Returns:
            Initialized Language Model
        """
        if llm_model == "OpenAI GPT-3.5":
            # Requires OPENAI_API_KEY
            return OpenAI(temperature=temperature)
        
        # Hugging Face models
        model_name = self.LLM_MODELS.get(llm_model, "microsoft/DialoGPT-small")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=200,
            temperature=temperature,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def create_chat_chain(self, llm):
        """
        Create conversational retrieval chain
        
        Args:
            llm: Initialized Language Model
        """
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs=self.search_kwargs
            ),
            return_source_documents=True
        )
    
    def chat_with_documents(
        self, 
        message: str, 
        history: List[List[str]]
    ) -> str:
        """
        Handle chat interaction with document context
        
        Args:
            message (str): User's current message
            history (List): Conversation history
        
        Returns:
            str: AI's response
        """
        # Prepare chat history
        chat_history = [(h[0], h[1]) for h in history]
        
        try:
            # Query the conversational chain
            result = self.chat_chain({
                'question': message,
                'chat_history': chat_history
            })
            
            response = result['answer']
            
            # Optional: Print source documents
            print("Source Documents:")
            for doc in result['source_documents']:
                print(doc.page_content)
            
            return response
        
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def launch_gradio_interface(self):
        """
        Launch Gradio interface with customization options
        """
        with gr.Blocks() as demo:
            # Customization Components
            with gr.Row():
                embedding_dropdown = gr.Dropdown(
                    choices=list(self.EMBEDDING_MODELS.keys()),
                    value="HuggingFace MiniLM",
                    label="Embedding Model"
                )
                llm_dropdown = gr.Dropdown(
                    choices=list(self.LLM_MODELS.keys()),
                    value="DialoGPT Small",
                    label="Language Model"
                )
            
            with gr.Row():
                chunk_size = gr.Slider(
                    minimum=100, 
                    maximum=2000, 
                    value=1000, 
                    label="Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0, 
                    maximum=500, 
                    value=200, 
                    label="Chunk Overlap"
                )
                temperature = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.7, 
                    label="Model Temperature"
                )
                k_documents = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=3, 
                    label="Retrieved Documents"
                )
            
            # Process and Initialize Button
            initialize_btn = gr.Button("Initialize Chat")
            
            # Chat Interface
            chatbot = gr.ChatInterface(
                fn=self.chat_with_documents,
                title="Customizable Document Chat"
            )
            
            # Initialize Logic
            initialize_btn.click(
                fn=self._initialize_from_config,
                inputs=[
                    embedding_dropdown, 
                    llm_dropdown, 
                    chunk_size, 
                    chunk_overlap, 
                    temperature, 
                    k_documents
                ],
                outputs=chatbot
            )
            
        demo.launch()
    
    def _initialize_from_config(
        self, 
        embedding_model, 
        llm_model, 
        chunk_size, 
        chunk_overlap, 
        temperature, 
        k_documents
    ):
        """
        Initialize application with user-specified configuration
        """
        # Update configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_kwargs = {"k": int(k_documents)}
        
        # Load and process documents
        texts = self.load_and_process_documents()
        
        # Create vector store
        self.create_vector_store(texts, embedding_model)
        
        # Initialize LLM
        llm = self.initialize_llm(llm_model, temperature)
        
        # Create chat chain
        self.create_chat_chain(llm)
        
        return gr.ChatInterface(fn=self.chat_with_documents)

# Main execution
if __name__ == '__main__':
    app = DocumentChatApp()
    app.launch_gradio_interface()