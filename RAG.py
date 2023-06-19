import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from langchain.llms import HuggingFacePipeline

# model for embeddings
model_name = "/home/kali/dev/models/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
#encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)
# encode_kwargs=encode_kwargs

#loader = PyPDFLoader("/home/kali/dev/blackrock_weekly_investment_commentary.pdf")
#pages = loader.load_and_split()
#print("len pages : ", len(pages))
#print(pages[0])

# advanced method : split by chunk
import textract
doc = textract.process("/home/kali/dev/wimbledon.pdf")

with open("/home/kali/dev/wimbledon.pdf", "w") as f:
    f.write(doc.decode('utf-8'))

with open("/home/kali/dev/wimbledon.pdf", "r") as f:
    text = f.read()

# creation function to count tokens

tokenizer = GPT2TokenizerFast.from_pretrained("/home/kali/dev/models/gpt2")

def count_tokens(text:str) -> int:
    return len(tokenizer.encode(text, max_length=512))

# split text into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

#print("chunks : ", chunks)

# quick data viz to check if chunking was successful :
#token_counts = [count_tokens(chunk.page_content)for chunk in chunks]

#df = pd.DataFrame({"Token Count" : token_counts})

#df.hist(bins=40, )

#plt.show()
#model = GPT4All(model="/home/kali/dev/models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512)
model = HuggingFacePipeline.from_model_id(
    model_id="/home/kali/dev/models/flan-t5-base",
    task="text2text-generation",
    
)               
chain = load_qa_chain(model, chain_type="stuff")
db = FAISS.from_documents(chunks, embeddings)

query = "How many competitors were there in the first Wimbledon tournament?"
docs = db.similarity_search(query)

#print(docs[0])
#chain.run(input_documents=docs, question=query)
print(chain.run(input_documents=docs, question=query))
print(docs[0])
