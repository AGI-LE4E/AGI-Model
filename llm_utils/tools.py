import requests
import random
import pandas as pd

from collections import Counter
from typing import Annotated
from langchain_core.tools import tool

@tool
def query_restaurants(search_term: str) -> str:
    """Search restaurants"""
    df = pd.read_csv("./data/jeju_preprocessed.csv")
    nearby_restaurants = df[df["lodgment"].str.contains(search_term)]["nearby_restaurants"].to_list()
    return nearby_restaurants
