import math
from typing import Dict, Any
class FingerCounter:
    # Pulgar
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    # Índice
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    # Medio
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    # Anular
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    # Meñique
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    FINGER_EXTENSION_ANGLE = 155.0
    def __init__(self):
        self.last_result = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
    @staticmethod
    def distance(point_a, point_b):
        dx = point_a.x - point_b.x
        dy = point_a.y - point_b.y

        return math.sqrt(dx * dx + dy * dy)
    @staticmethod
    def angle(point_a, point_b, point_c):
        vector_ba = (point_a.x - point_b.x,point_a.y - point_b.y)
        vector_bc = (point_c.x - point_b.x,point_c.y - point_b.y)
        magnitude_ba = math.sqrt(vector_ba[0] ** 2 + vector_ba[1] ** 2)
        magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0
        dot_product = (vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1])
        cosine = (dot_product /(magnitude_ba * magnitude_bc))
        cosine = max(-1.0,min(1.0, cosine))
        return math.degrees(math.acos(cosine))
    def is_finger_extended(self,landmarks,mcp,pip,dip,tip):
        pip_angle = self.angle(landmarks[mcp],landmarks[pip],landmarks[dip])
        dip_angle = self.angle(landmarks[pip],landmarks[dip],landmarks[tip])
        return (pip_angle >= self.FINGER_EXTENSION_ANGLE and dip_angle >= self.FINGER_EXTENSION_ANGLE)
    def is_thumb_extended(self,landmarks):
        mcp_angle = self.angle(landmarks[self.THUMB_CMC],landmarks[self.THUMB_MCP],landmarks[self.THUMB_IP])
        ip_angle = self.angle(landmarks[self.THUMB_MCP],landmarks[self.THUMB_IP],landmarks[self.THUMB_TIP])
        thumb_distance = self.distance(landmarks[self.THUMB_TIP],landmarks[0])
        base_distance = self.distance(landmarks[self.THUMB_MCP],landmarks[0])
        return (mcp_angle >= 140.0 and ip_angle >= 140.0 and thumb_distance > base_distance * 1.25)
    def analyze(self,landmarks) -> Dict[str, bool]:
        result = {"thumb": self.is_thumb_extended(landmarks),"index": self.is_finger_extended(landmarks,self.INDEX_MCP,self.INDEX_PIP,self.INDEX_DIP,self.INDEX_TIP),"middle": self.is_finger_extended(landmarks,self.MIDDLE_MCP,self.MIDDLE_PIP,self.MIDDLE_DIP,self.MIDDLE_TIP),"ring": self.is_finger_extended(landmarks,self.RING_MCP,self.RING_PIP,self.RING_DIP,self.RING_TIP),"pinky": self.is_finger_extended(landmarks,self.PINKY_MCP,self.PINKY_PIP,self.PINKY_DIP,self.PINKY_TIP)}
        self.last_result = result
        return result
    @staticmethod
    def count(fingers: Dict[str, bool]) -> int:
        return sum(fingers.values())
    @staticmethod
    def pattern(fingers: Dict[str, bool]) -> str:
        names = [name.upper() for name, extended in fingers.items() if extended]
        if not names:
            return "NONE"
        return " + ".join(names)