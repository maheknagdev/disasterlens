from pydantic import BaseModel, Field

SEVERITY_LEVELS = ["none", "mild", "moderate", "severe"]


class ResourceNeeds(BaseModel):
    food: int = Field(ge=1, le=5)
    water: int = Field(ge=1, le=5)
    shelter: int = Field(ge=1, le=5)
    medical: int = Field(ge=1, le=5)


class FusionOutput(BaseModel):
    final_severity: str = Field(description="One of: none, mild, moderate, severe")
    resource_needs: ResourceNeeds
    priority_summary: str = Field(description="1-2 sentence natural-language relief priority summary")
