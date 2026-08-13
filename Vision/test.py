from pathlib import Path
import mediapipe as mp


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Models"
    / "hand_landmarker.task"
)


# ============================================================
# INFORMACIÓN
# ============================================================

print("=" * 60)
print("             MUSIC HANDS - MODEL TEST")
print("=" * 60)

print(f"MediaPipe: {mp.__version__}")
print(f"Modelo:    {MODEL_PATH}")
print(f"Existe:    {MODEL_PATH.exists()}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No existe el modelo:\n{MODEL_PATH}"
    )

print(f"Tamaño:    {MODEL_PATH.stat().st_size:,} bytes")


# ============================================================
# LEER MODELO COMO BYTES
# ============================================================

print("\nLeyendo modelo...")

model_buffer = MODEL_PATH.read_bytes()

print(
    f"Bytes cargados: {len(model_buffer):,}"
)


# ============================================================
# CONFIGURAR MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_buffer=model_buffer
    ),

    num_hands=2
)


# ============================================================
# CREAR DETECTOR
# ============================================================

print("\nIntentando crear HandLandmarker...")

try:

    detector = (
        HandLandmarker.create_from_options(
            options
        )
    )

    print("\n" + "=" * 60)
    print("✅ HAND LANDMARKER CARGADO CORRECTAMENTE")
    print("=" * 60)

    detector.close()

except Exception as error:

    print("\n" + "=" * 60)
    print("❌ ERROR AL CARGAR HAND LANDMARKER")
    print("=" * 60)

    print(f"\nTipo: {type(error).__name__}")
    print(f"Mensaje: {error}")

    raise