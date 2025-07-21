from pydantic import BaseModel
from typing import Optional, Union, List

class Resolution(BaseModel):
    height: int
    width: int

class EditObject(BaseModel):
    class_name: str
    bbox: list[float]
    mask: str
    score: Union[list[float], float]
    clip_score: float
    aes_score: float

class ReplaceObjectTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj: EditObject
    edit_type: str
    edit_prompt: str
    edit_result: str

class RemoveObjectTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj: EditObject
    edit_type: str
    edit_prompt: str
    edit_result: Optional[str] = None

class AddObjectTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj: Optional[EditObject] = None
    edit_type: str
    edit_prompt: str
    edit_result: EditObject

class BackgroundChangeTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_type: str
    edit_prompt: str
    edit_result: str

class AdjustCannyTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj: EditObject
    edit_type: str
    edit_prompt: str
    edit_result: str
    
class ComposeTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj1: EditObject
    edit_obj2: EditObject
    edit_type: List[str]
    edit_prompt: Optional[str] = None
    round1_prompt: Optional[str] = None
    round2_prompt: Optional[str] = None
    edit_result1: str
    edit_result2: str

class OmitTask(BaseModel):
    original_path: str
    resolution: Resolution
    edit_obj: EditObject
    edit_type: List[str]
    round1_prompt: Optional[str] = None
    round2_prompt: Optional[str] = None
    round3_prompt: Optional[str] = None
    edit_result2: str
    edit_result3: str
