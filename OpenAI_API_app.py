# Bring in deps
import os
from dotenv import load_dotenv
from pathlib import Path

import streamlit as st
from langchain_openai.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Open AI key
dotenv_path = Path('/home/artem/Work/Best-Hackaton/.env')
load_dotenv(dotenv_path=dotenv_path)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# App framework
st.title('🔥 Helparik 🔥')
prompt = st.text_input('Please describe what you need')

categories_list = ['Medical', 'Military', 'Food Supply', 'Law', 'Psychological', 'Clothes', 'Children'
    , 'Household goods', 'Equipment and tools']

categories_string = ', '.join(categories_list)

# Prompt templates
category_template = PromptTemplate(
    input_variables=['description'],
    template='You have the next categories:' + categories_string
             + ' Which of the categories the best corresponds to the next description? ' + '{description}'
             + ' give the answer by one word'
)

advice_template = PromptTemplate(
    input_variables=['category', 'wikipedia_research'],
    template='Give me the advice on that category: CATEGORY: {category}'
             + ' take in mind you are giving the advice for people from Ukraine and leveraging this wikipedia'
               'research:{wikipedia_research}'
)

# Memory
category_memory = ConversationBufferMemory(input_key='description', memory_key='chat_history')
advice_memory = ConversationBufferMemory(input_key='category', memory_key='chat_history')

# Llms chains
llm = OpenAI(temperature=0.9)
category_chain = LLMChain(llm=llm, prompt=category_template, verbose=True
                             , output_key='category', memory=category_memory)
advice_chain = LLMChain(llm=llm, prompt=advice_template, verbose=True
                          , output_key='advice', memory=advice_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen
if prompt:
    category = category_chain.run(description=prompt)
    wiki_research = wiki.run(prompt)
    advice = advice_chain.run(category=category, wikipedia_research=wiki_research)

    st.write(category)
    st.write(advice)

    with st.expander('Category History'):
        st.info(category_memory.buffer)

    with st.expander('Advice History'):
        st.info(advice_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(advice_memory.buffer)