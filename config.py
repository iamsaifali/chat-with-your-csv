import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    @classmethod
    def set_api_key(cls, api_key):
        cls.OPENAI_API_KEY = api_key
        os.environ["OPENAI_API_KEY"] = api_key

def get_api_key():
    return Config.OPENAI_API_KEY

def set_api_key(api_key):
    Config.set_api_key(api_key)