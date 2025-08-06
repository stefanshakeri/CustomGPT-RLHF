#!/usr/bin/env python3
"""
Script to compare expert and LLM responses in a Chroma vector database through cosine similarity. 
"""

import os
import argparse

import openai
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from create_database import OPENAI_API_KEY, CHROMA_PATH

# load environment variables
load_dotenv()

OUTPUT_PATH = "data/" + os.getenv("OUTPUT_FILE", "comparison_results.csv")

openai.api_key = OPENAI_API_KEY


def prepare_db() -> Chroma:
    """
    Prepare the database for querying. 

    :return: Chroma database instance
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )
    return db


def get_expert_responses(db: Chroma) -> list[str]:
    """
    Retrieve expert responses from the Chroma database and 
    return them as a list. 

    :param db: Chroma database instance
    :return: list of expert responses
    """

    # Query the database for metadata where "type" = "expert"
    expert_docs = db.get(
        where={"type": "expert"}
    )
    if not expert_docs:
        raise ValueError("No expert responses found in the database.")
    
    expert_list = expert_docs["documents"]

    return expert_list


def find_similar_responses(expert_response: str, db: Chroma) -> list[tuple[Document, float]]:
    """
    Find the two most similar LLM responses to a given expert response. 

    :param expert_response: the expert response to compare to
    :param db: Chroma database instance
    :return: list of the two most similar LLM responses and their similarity scores
    """
    results = db.similarity_search_with_relevance_scores(
        expert_response, k=2, filter={"type": "llm"})
    
    #print(f"similarity search results: {results}")

    return results


def add_to_dataframe(expert_response: str, similar_responses: list[tuple[Document, float]], df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the expert response and the two most similar LLM responses (and their similarity scores) to the DataFrame.
    
    :param expert_response: the expert response to add
    :param similar responses: list of the two most similar LLM responses and their similarity scores
    :param df: DataFrame to add the results to
    """
    if len(similar_responses) < 2:
        print("Not enough similar responses found.")
        return
    
    llm_response_1, score_1 = similar_responses[0]
    llm_response_2, score_2 = similar_responses[1]

    df = pd.concat([df, pd.DataFrame([{
        "expert_response": expert_response,
        "llm_response_1": llm_response_1.page_content,
        "llm_response_2": llm_response_2.page_content,
        "similarity_1": score_1,
        "similarity_2": score_2
    }])], ignore_index=True)

    print(f"Added to DataFrame: {expert_response[:50]}...\n\tresponse 1: {llm_response_1.page_content[:50]}...\n\tresponse 2: {llm_response_2.page_content[:50]}...\n\tscore 1: {score_1}\n\tscore 2: {score_2}")

    return df

def main():
    """
    Main function to compare expert and LLM responses.
    """
    # get the expert responses from the database
    db = prepare_db()
    expert_list = get_expert_responses(db)

    # for each expert response, find the two most similar LLM responses
    comparison_results = pd.DataFrame(columns=["expert_response", "llm_response_1", "llm_response_2", "similarity_1", "similarity_2"])
    for expert_response in expert_list:
        print(f"Comparing expert response: {expert_response[:50]}...")
        similar_responses = find_similar_responses(expert_response, db)

        comparison_results = add_to_dataframe(expert_response, similar_responses, comparison_results)

    # output results into a CSV file
    comparison_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Comparison results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()