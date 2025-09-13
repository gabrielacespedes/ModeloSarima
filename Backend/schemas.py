from pydantic import BaseModel
from typing import List, Dict

class VentasRequest(BaseModel):
    data: List[Dict]
