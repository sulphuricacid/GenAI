import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

#imporing necessary packages packages from langchain
# from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableMap
from operator import itemgetter


RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

text =  "I am a biology medical book"
mcq_count = 2
subject= "Biology"
tone= "simple"


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
OPENAPIk=os.getenv("OPENAI_API_KEY")
print(OPENAPIk)

llm = OpenAI(openai_api_key=OPENAPIk,model_name="gpt-3.5-turbo-instruct", temperature=0.7)

template="""Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template)


quiz_chain= quiz_generation_prompt | llm | StrOutputParser()


template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template2)


review_chain = quiz_evaluation_prompt | llm |  StrOutputParser()
quiz_evaluation= quiz_evaluation_prompt | llm | StrOutputParser()

def merge_subject_and_quiz(data) -> dict:
    """all_inputs is the original input dict,
    quiz_output is the string returned by quiz_chain"""
    # print(data)
    return {
        "subject": data['sub'],
        "quiz": data['quiz_output']
    }


chain_with_pass = RunnableMap({
    "quiz_output": quiz_chain,        # This runs the LLMChain
    "sub": lambda x: x["subject"]   # Forward 'var2' for downstream use
})

generate_evaluate_chain = chain_with_pass | RunnableLambda(merge_subject_and_quiz) | quiz_evaluation

# generate_evaluate_chain = (
#     quiz_chain # 1) The quiz (string) from the LLM
#     | (lambda quiz_output: {"subject": inputs["subject"], "quiz": quiz_output})
#     | quiz_evaluation_prompt
#     | llm
#     | StrOutputParser()   # 2) The final evaluation as a string
# )


generate_evaluate_chain.invoke( {"text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)})
