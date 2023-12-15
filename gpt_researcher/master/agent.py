import time, os
from gpt_researcher.config import Config
from gpt_researcher.master.functions import *
from gpt_researcher.context.compression import ContextCompressor
from gpt_researcher.memory import Memory
from gpt_researcher.memory.embeddings import FAISS, OpenAIEmbeddings
import yfinance as yf
from datetime import datetime, timedelta
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from requests.exceptions import HTTPError
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

class GPTResearcher:
    """
    GPT Researcher
    """
    def __init__(self, query, report_type, config_path=None, websocket=None):
        """
        Initialize the GPT Researcher class.
        Args:
            query:
            report_type:
            config_path:
            websocket:
        """
        self.query = query
        self.agent = None
        self.role = None
        self.report_type = report_type
        self.websocket = websocket
        self.cfg = Config(config_path)
        self.retriever = get_retriever(self.cfg.retriever)
        self.context = []
        self.memory = Memory()
        self.visited_urls = set()
        self.dir_path= None
    async def run(self):
        """
        Runs the GPT Researcher
        Returns:
            Report
        """
        # Insert document formats
        print(f"🔎 귀사의 리포트 양식을 조사중에 있습니다")
        dir_path = 'frontend/static/hana'
        pdf_docs = await self.get_pdf_list(dir_path)
        raw_text = await self.get_pdf_text(pdf_docs)
        text_chunks = await self.get_text_chunks(raw_text)
        vectorstore = await self.get_vectorstore(text_chunks)
        llm_papers = await self.get_retriever_chain(vectorstore)

        tools = [Tool(
        name = "Document Store",
        func = llm_papers.run,
        description = "Useful to lookup information from documents given",
#        handle_tool_error=True
    ),
]
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key="")
        document_agent = initialize_agent(tools=tools,
                  llm=llm,
                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        self.report_format=document_agent.run(f'please return me common features of documents uploaded in terms of layout, \
                                              structure and contents-wise. this could be anything such as target price is a must,\
                                               stock name is always followed by its ticker, it always mentions\
                                              about the industry competition and so forth.\
                                              Try YOUR BEST to get the result than deter to apologizing me for failing to do your job')

        
        print(f"🔎 Running research for '{self.query}'...")
        # Generate Agent
        self.agent, self.role = await choose_agent(self.query, self.cfg)
        await stream_output("logs", self.agent, self.websocket)

        # Generate Sub-Queries including original query
        sub_queries = await get_sub_queries(self.query,self.report_format, self.role, self.cfg)
        await stream_output("logs",
                            f"🧠 I will conduct my research based on the following queries: {sub_queries}...",
                            self.websocket)

        # Run Sub-Queries
        for sub_query in sub_queries:
            await stream_output("logs", f"\n🔎 Running research for '{sub_query}'...", self.websocket)
            scraped_sites = await self.scrape_sites_by_query(sub_query)
            context = await self.get_similar_content_by_query(sub_query, scraped_sites)
            await stream_output("logs", f"📃 {context}", self.websocket)
            self.context.append(context)

         # Generate the stock ticker symbol using LLM
        search = TavilySearchAPIWrapper()
        tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
        tools = [Tool(
                name="Search",
                func=tavily_tool.run,
                description="useful for getting ticker"
            )
        ]
        
        ticker_agent = initialize_agent(tools=tools,
                          llm=llm,
                          agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        stock_ticker = ticker_agent.run(f'return me the stock ticker from {self.query}\
                                        please return only TICKER(such as APPL for apple \
                                        or 005930.KS for samsung electronics), NO STRINGS ATTACHED.\
                                        do not include any quotes surrounding ticker because the\
                                        returned ticker will be directly used as parameter for yf.download')
        

        # Insert current stock data
        stock_data = yf.download(stock_ticker, start="2022-01-01", end=datetime.today())
        self.context.append(stock_data)

        

        # Conduct Research
        await stream_output("logs", f"✍️ Writing {self.report_type} for research task: {self.query}...", self.websocket)
        report = await generate_report(query=self.query, context=self.context, report_format=self.report_format,
                                       agent_role_prompt=self.role, report_type=self.report_type,
                                       websocket=self.websocket, cfg=self.cfg)
        time.sleep(2)
        return report

    async def get_new_urls(self, url_set_input):
        """ Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.visited_urls:
                await stream_output("logs", f"✅ Adding source url to research: {url}\n", self.websocket)

                self.visited_urls.add(url)
                new_urls.append(url)

        return new_urls

    async def scrape_sites_by_query(self, sub_query):
        """
        Runs a sub-query
        Args:
            sub_query:

        Returns:
            Summary
        """
        # Get Urls
        retriever = self.retriever(sub_query)
        try:
            search_results = retriever.search(max_results=self.cfg.max_search_results_per_query)
        except HTTPError as e:
            # Handle the 502 error
            print("Error: Failed to retrieve search results. Skipping this part.")
            search_results =[]
        new_search_urls = await self.get_new_urls([url.get("href") for url in search_results])

        # Scrape Urls
        # await stream_output("logs", f"📝Scraping urls {new_search_urls}...\n", self.websocket)
        await stream_output("logs", f"🤔Researching for relevant information...\n", self.websocket)
        scraped_content_results = scrape_urls(new_search_urls, self.cfg)
        return scraped_content_results

    async def get_similar_content_by_query(self, query, pages):
        await stream_output("logs", f"🌐 Summarizing url: {query}", self.websocket)
        # Summarize Raw Data
        context_compressor = ContextCompressor(documents=pages, embeddings=self.memory.get_embeddings())
        # Run Tasks
        return context_compressor.get_context(query, max_results=8)
    
    async def get_pdf_text(self, pdf_docs):
        text = "" #init
        for i in range(len(pdf_docs)):
            doc_text = PyMuPDFLoader(pdf_docs[i]).load()
            for j in range(len(doc_text)):
                text += doc_text[j].page_content
        print(text)
        return text

    
    async def get_text_chunks(self,text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    async def get_vectorstore(self, text_chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    
    async def get_retriever_chain(self, vectorstore):
        llm = ChatOpenAI()

        retriever_chain= RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever())
        return retriever_chain

    async def get_pdf_list(self, dir_path):
        pdf_docs = []
        for filename in os.listdir(dir_path):
         # Check if the file is a PDF
            if filename.endswith('.pdf'):
                # Construct full file path
                file_path = os.path.join(dir_path, filename)
                pdf_docs.append(file_path)
        
        return pdf_docs
