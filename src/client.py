import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek-v4-flash"

client = Anthropic(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
)