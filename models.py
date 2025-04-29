from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId # Import ObjectId

# Helper for Pydantic to handle ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class TextInput(BaseModel):
    text: str
    session_id: Optional[str] = None

class ProcessResponse(BaseModel):
    # Store ID as string in response model for simplicity
    interaction_id: str # Changed from int
    status: str
    message: str
    transcribed_text: Optional[str] = None
    llm_response: Optional[str] = None
    audio_output_path: Optional[str] = None
    video_output_path: Optional[str] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str} # Ensure ObjectId is serialized as string

class HealthCheckResponse(BaseModel):
    status: str = "ok"
    database_connected: bool
    whisper_loaded: bool = False
    tts_loaded: bool = False
    lipsync_loaded: bool = False