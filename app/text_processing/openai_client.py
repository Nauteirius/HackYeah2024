import os

from openai import OpenAI

instance = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
