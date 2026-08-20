from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
@dataclass
class HandData:
    label: str
    confidence: float # Confianza de MediaPipe en la clasificación.
    landmarks: List[Any] = field(default_factory=list)
    fingers: Dict[str, bool] = field(default_factory=lambda: {"thumb": False,"index": False,"middle": False,"ring": False,"pinky": False})
    finger_count: int = 0
    gesture: str = "UNKNOWN"
    angle: Optional[float] = None
    def update_finger_count(self) -> int:
        self.finger_count = sum(self.fingers.values())
        return self.finger_count

    def get_finger_names(self) -> List[str]:
        return [finger for finger, is_up in self.fingers.items() if is_up]
    def __str__(self) -> str:
        return (f"HandData(label={self.label}, confidence={self.confidence:.2f}, fingers={self.finger_count}, gesture={self.gesture}, angle={self.angle})")
    