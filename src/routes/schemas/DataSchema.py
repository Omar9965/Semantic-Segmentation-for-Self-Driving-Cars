from pydantic import BaseModel
from typing import List, Optional

class SegmentationResult(BaseModel):
    """Result for a single semantic segmentation"""
    filename: str                 
    original_image_url: str        
    mask_url: str                 
    overlay_url: Optional[str]     
    width: int
    height: int


class MultipleSegmentationResponse(BaseModel):
    """Response for multiple semantic segmentation"""
    results: List[SegmentationResult]