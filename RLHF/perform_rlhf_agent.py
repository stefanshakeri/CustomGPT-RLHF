#!usr/bin/env python3
"""
Script to use a GPT-4o model as RLHF to modify a Custom GPT prompt. 
Uses the CSV file created through the agent.compare_with_llm script to compare the expert response with the LLM responses.
"""

import os

import openai
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI

from vector.create_database import OPENAI_API_KEY
from agent.compare_with_llm import OUTPUT_PATH, GPT_MODEL

# load environment variables
load_dotenv()

INPUT_TXT = "data/" + os.getenv("INPUT_PROMPT", "input_prompt.txt")
OUTPUT_TXT = "data/" + os.getenv("OUTPUT_PROMPT_AGENT", "output_prompt_agent.txt")

INPUT_CSV = OUTPUT_PATH
openai.api_key = OPENAI_API_KEY

# initialize the GPT-4o model
client = OpenAI()

prompt_modifier_text = '''
You are an expert agent in performing RLHF on a Custom GPT prompt. Your task is to analyze a sample response from the custom GPT
compared to an expert response, along with feedback from an LLM judge on why that response is most similar to the expert response,
as well as the prompt for the Custom GPT. Your goal is to modify the prompt to improve the Custom GPT's response, using the provided context. 
Keep the prompt in Markdown format. 

Expert Response: {expert_response}
LLM Response: {llm_response}
LLM Judge Feedback: {llm_judge_feedback}
Current Prompt: {current_prompt}
'''

better_response_evaluator_text = '''
You are an expert in evaluating LLM responses. Your task is to analyze the feedback from an LLM judge on the similarity of a Custom GPT response
to an expert response, and determine if the LLM judge decided if prompt 1 or prompt 2 is better. 
Your goal is to output only one number, either "1" if prompt 1 is better, or "2" if prompt 2 is better.
The output must be just the number, with no additional text or formatting.

LLM Judge Feedback: {llm_judge_feedback}
'''


def get_all_responses() -> list[tuple[str, str, str, str]]:
    """
    Reads the CSV file in INPUT_CSV and extracts expert response, LLM responses, and LLM judge feedback.

    :return: a list of tuples containing expert response, LLM responses, and LLM judge feedback
    """

    df = pd.read_csv(INPUT_CSV)

    #print(f"Example row from CSV: {df.iloc[0].to_dict()}")

    if 'expert_response' not in df.columns or 'llm_response_1' not in df.columns or 'llm_response_2' not in df.columns or 'comparison' not in df.columns:
        raise ValueError(f"Required columns not found in {INPUT_CSV}. Expected columns: 'expert_response', 'llm_response_1', 'llm_response_2', 'comparison'.")
    
    responses = []

    for _, row in df.iterrows():
        expert_response = row['expert_response']
        llm_response_1 = row['llm_response_1']
        llm_response_2 = row['llm_response_2']
        llm_judge_feedback = row['comparison']

        responses.append((expert_response, llm_response_1, llm_response_2, llm_judge_feedback))

    if not responses:
        raise ValueError("No valid responses found in the CSV file.")
    
    return responses


def determine_better_response(llm_judge_feedback: str) -> int:
    """
    Determine which prompt is better based on the LLM judge feedback.

    :param llm_judge_feedback: the feedback from the LLM judge
    :return: 1 if prompt 1 is better, 2 if prompt 2 is better
    """

    # prompt the GPT-4o model to evaluate the feedback
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "user", "content": better_response_evaluator_text.format(llm_judge_feedback=llm_judge_feedback)}
        ]
    )

    # extract the response text
    response_text = response.choices[0].message.content.strip()

    if response_text not in ["1", "2"]:
        raise ValueError(f"Unexpected response from GPT-4o model: {response_text}. Expected '1' or '2'.")
    
    return int(response_text)





def main():
    """
    Main function to read responses from CSV and determine which prompt is better.
    """
    responses = get_all_responses()

    better_response_num = determine_better_response(responses[0][3])  # Use the first response's judge feedback for demonstration

    print(f"Better response is from prompt: {better_response_num}")


if __name__ == "__main__":
    main()