from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class LLMResponse(BaseModel):
    text: str = Field(description="The generated text.")
    usage: Dict[str, Any] = Field(description="Usage information for the request")
    stop_reason: Optional[str] = Field(
        description="Reason for stopping the generation."
    )
