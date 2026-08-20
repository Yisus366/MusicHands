from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
@dataclass
class HandData:
    # ========================================================
    # IDENTIFICACIÓN DE LA MANO
    # ========================================================
    # "Left" o "Right"
    label: str
    # Confianza de MediaPipe en la clasificación.
    # Ejemplo: 0.97
    confidence: float
    # ========================================================
    # LANDMARKS
    # ========================================================
    # Lista de los 21 landmarks proporcionados por MediaPipe.
    # Índices importantes:
    # 0  = muñeca
    # 4  = punta del pulgar
    # 8  = punta del índice
    # 12 = punta del medio
    # 16 = punta del anular
    # 20 = punta del meñique

    landmarks: List[Any] = field(default_factory=list)
    # ========================================================
    # ESTADO DE LOS DEDOS
    # ========================================================
    # True  → dedo levantado
    # False → dedo cerrado
    # Todavía NO calculamos estos valores.
    # Eso será responsabilidad de finger_counter.py.
    fingers: Dict[str, bool] = field(default_factory=lambda: {"thumb": False,"index": False,"middle": False,"ring": False,"pinky": False})
    # ========================================================
    # CANTIDAD DE DEDOS
    # ========================================================
    # Cantidad total de dedos levantados.
    # Inicialmente es 0 porque todavía no hemos creado
    # el contador de dedos.
    finger_count: int = 0

    # ========================================================
    # GESTO
    # ========================================================

    # Nombre del gesto detectado.
    # Por ahora no tenemos detector de gestos,
    # por eso comienza como UNKNOWN.

    gesture: str = "UNKNOWN"

    # ========================================================
    # ORIENTACIÓN DE LA MANO
    # ========================================================
    # Ángulo de inclinación de la mano en grados.
    # Todavía no lo calculamos.
    # None significa:
    # "todavía no disponible".
    angle: Optional[float] = None


    # ========================================================
    # MÉTODO: ACTUALIZAR CANTIDAD DE DEDOS
    # ========================================================
    def update_finger_count(self) -> int:
        """
        Cuenta los dedos que actualmente están marcados
        como levantados.
        """
        self.finger_count = sum(self.fingers.values())
        return self.finger_count
    # ========================================================
    # MÉTODO: OBTENER NOMBRES DE DEDOS LEVANTADOS
    # ========================================================

    def get_finger_names(self) -> List[str]:
        """
        Devuelve una lista con los nombres de los dedos
        actualmente levantados.
        """

        return [
            finger
            for finger, is_up in self.fingers.items()
            if is_up
        ]

    def __str__(self) -> str:
        """
        Permite imprimir el objeto de forma legible.
        """

        return (
            f"HandData("
            f"label={self.label}, "
            f"confidence={self.confidence:.2f}, "
            f"fingers={self.finger_count}, "
            f"gesture={self.gesture}, "
            f"angle={self.angle}"
            f")"
        )
    