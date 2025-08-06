#!/usr/bin/env python3
"""
Script to compare expert and LLM responses in a Chroma vector database through cosine similarity. 
"""

import os
import argparse

import openai
import pandas as pd
from dotenv import load_dotenv

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
    