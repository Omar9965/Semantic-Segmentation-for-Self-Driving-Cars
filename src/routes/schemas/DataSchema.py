from pydantic import BaseModel
from typing import List, Optional, Dict

class SegmentationResult(BaseModel):
    """Result for a single semantic segmentation"""
    filename: str                 
    original_image_url: str        
    mask_url: str                 
    overlay_url: Optional[str]     
    width: int
    height: int
    detected_classes: List[str]           # List of detected class names
    class_counts: Dict[str, int]          # Pixel count per class
    has_vehicles: bool                    # Whether vehicles are detected
    has_pedestrians: bool                 # Whether pedestrians are detected
    has_road: bool                        # Whether road is detected


class MultipleSegmentationResponse(BaseModel):
    """Response for multiple semantic segmentation"""
    results: List[SegmentationResult]