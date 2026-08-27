import math
class HandGeometry:
    WRIST = 0

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        self.last_angle = 0.0

    @staticmethod
    def vector(point_a, point_b):
        return (point_b.x - point_a.x,point_b.y - point_a.y)
    
    @staticmethod
    def vector_angle(vector):
        x, y = vector
        angle = math.degrees(math.atan2(y, x))
        return angle
    
    def palm_angle(self, landmarks):
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        palm_vector = self.vector(wrist,middle_mcp)
        angle = self.vector_angle(palm_vector)
        self.last_angle = angle
        return angle
    
    @staticmethod
    def normalize_angle(angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def musical_angle(self, landmarks):
        raw_angle = self.palm_angle(landmarks)
        angle = raw_angle - 90.0

        angle = self.normalize_angle(angle)
        return angle