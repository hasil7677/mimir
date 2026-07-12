from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = "user"  # user | assistant | tool | system
    content: str


class CaptureRequest(BaseModel):
    # No tenant_id field on purpose — tenant identity comes from auth only.
    user_id: str
    session_id: str
    messages: list[Message] = Field(min_length=1)


class SessionEndRequest(BaseModel):
    user_id: str
    session_id: str


class RecallRequest(BaseModel):
    user_id: str
    query: str
    session_id: str | None = None  # supplied -> recent hot turns are injected
