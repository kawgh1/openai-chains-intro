from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default='python')
args = parser.parse_args()

# Secure this key!
# api_key="api key"

# # this code is replaced by using a .env file with the API key
# # and pip install python-dotenv --> load_dotenv()
# llm = OpenAI(
#     openai_api_key=api_key
# )

llm = OpenAI()

# First Chain Prompt
code_prompt = PromptTemplate(
    template="Write a very short {language} " +
        "function that will {task}",
    input_variables=['language', 'task']
)

# First Chain to generate code
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# Second Chain Prompt
test_prompt = PromptTemplate(
    input_variables=['language', 'code'],
    template="Write a code test for the following" +
    "{language} code:\n{code}"
)

# Second Chain to check the code
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

sequentialChain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)


result = sequentialChain({
    "language": args.language,
    "task": args.task
})

print(">>>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>>> GENERATED TEST:")
print(result['test'])