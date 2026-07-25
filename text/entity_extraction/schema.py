from typing import Literal
from pydantic import BaseModel, Field

ResourceType = Literal["food", "water", "shelter", "medical"]


class EntityExtraction(BaseModel):
    population_estimate: int | None = Field(
        description="Estimated number of people affected, mentioned or clearly implied in the text. Null if not stated."
    )
    resource_types_mentioned: list[ResourceType] = Field(
        description="Which of food/water/shelter/medical are explicitly mentioned or implied as needed."
    )
    locations: list[str] = Field(
        description="Named location entities mentioned (city, neighborhood, landmark, region)."
    )
