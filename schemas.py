from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Section:
    content: str
    section_title: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Document:
    content: str
    metadata: Dict = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)


@dataclass
class Chunk:
    content: str
    metadata: Dict = field(default_factory=dict)