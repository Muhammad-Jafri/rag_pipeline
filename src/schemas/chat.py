from pydantic import BaseModel


class Chat(BaseModel):
    query: str
