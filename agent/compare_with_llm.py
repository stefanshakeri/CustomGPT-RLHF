#!/usr/bin/env python3
"""
Script to use an GPT-4o model agent acting as a judge to compare the similarity in content
of the two LLM responses to the expert response to determine which is more similar.
"""

import os

import openai
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from langchain.schema import Document

from vector.create_database import OPENAI_API_KEY, get_expert_responses
from vector.add_llm_responses import get_llm_responses

# load environment variables
load_dotenv()

OUTPUT_PATH = "data/" + os.getenv("OUTPUT_CSV", "llm_comparison_results.csv")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

openai.api_key = OPENAI_API_KEY

# initialize the GPT-4o model
client = OpenAI()

# define the input text for the judge agent
input_text = '''
You are an expert judge tasked with comparing two LLM responses to an expert response. 
Evaluate the similarity of each LLM response to the expert response and determine which LLM response is more similar.
Do not evaluate the similarity of the LLM responses to the expert response based on syntax or length of response, only on the content and meaning of the response.

Expert Response: {expert_response}

LLM Response 1: {llm_response_1}
LLM Response 2: {llm_response_2}
Compare the two LLM responses to the expert response and determine which is more similar.
'''

def extract_responses() -> list[tuple[str, str, str]]:
    """
    Extract the expert and LLM responses from the input CSV file and groups them by question ID. 

    :return: list of tuples containing expert response, LLM response 1, and LLM response 2
    """
    expert_docs = get_expert_responses()
    llm_docs = get_llm_responses()

    if not expert_docs or not llm_docs:
        raise ValueError("No expert or LLM responses found.")
    
    grouped_responses = []
    
    # Group LLM responses by question ID
    for expert_doc in expert_docs:
        question_id = expert_doc.metadata["question_id"]
        llm_responses = [doc for doc in llm_docs if doc.metadata["question_id"] == question_id]

        if not llm_responses:
            print(f"No LLM responses found for question ID {question_id}. Skipping.")
            continue

        grouped_responses.append((expert_doc.page_content, 
                                  llm_responses[0].page_content, 
                                  llm_responses[1].page_content))
        
    return grouped_responses


def add_to_csv(expert_response: str, llm_response_1: str, llm_response_2: str, comparison: str):
    """
    Append the comparison results to the output CSV file.

    :param expert_response: the expert response
    :param llm_response_1: the first LLM response
    :param llm_response_2: the second LLM response
    :param comparison: the comparison result from the judge agent
    """
    df = pd.DataFrame([{
        "expert_response": expert_response,
        "llm_response_1": llm_response_1,
        "llm_response_2": llm_response_2,
        "comparison": comparison
    }])

    if os.path.exists(OUTPUT_PATH):
        df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_PATH, index=False)

    print(f"Added to CSV: {expert_response[:50]}...\n\tresponse 1: {llm_response_1[:50]}...\n\tresponse 2: {llm_response_2[:50]}...\n\tcomparison: {comparison[:50]}...")


def compare_responses(responses: list[tuple[str, str, str]]):
    """
    Compare the expert response with the two LLM responses using the judge agent.

    :param responses: list of tuples containing expert response, LLM response 1, and LLM response 2
    """
    # Iterate through each response and get the comparison from the GPT-4o model
    if not responses:
        print("No responses to compare.")
        return
    
    for i, (expert_response, llm_response_1, llm_response_2) in enumerate(responses):
        comparison = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "user", "content": input_text.format(
                    expert_response=expert_response,
                    llm_response_1=llm_response_1,
                    llm_response_2=llm_response_2
                )}
            ]
        )

        if not comparison.choices or not comparison.choices[0].message.content:
            print(f"Error: No comparison result for response {i + 1}.")
            continue

        comparison_text = comparison.choices[0].message.content
        print(f"Comparison {i + 1}: {comparison_text[:50]}...")
        add_to_csv(expert_response, llm_response_1, llm_response_2, comparison_text)


def main():
    """
    Main function for running the LLM comparison script. 
    """
    responses = extract_responses()
    if not responses:
        print("No responses found to compare.")
        return
    
    compare_responses(responses)


if __name__ == "__main__":
    main()