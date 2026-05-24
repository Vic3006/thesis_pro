# -*- coding: utf-8 -*-
"""
================================================================================
save_models.py  —  Guarda tus modelos entrenados en el formato que espera el
                   dashboard OSTEO-AI (app.py).

Cómo usarlo:
  - Pégalo AL FINAL de tu notebook de Colab, DESPUÉS de entrenar tu mejor
    modelo tabular y tu mejor CNN.
  - Ajusta las 3 variables marcadas con  # <<< AJUSTA.
  - Ejecuta. Genera la carpeta models/ con los 3 archivos.
  - Descarga models/ y colócala junto a app.py. Pon DEMO_MODE=False.
================================================================================
"""
import os
import joblib

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------- TABULAR
# <<< AJUSTA: tu mejor clasificador tabular ya entrenado (ej. LogReg o RF).
#     Debe haber sido entrenado con las 22 features en el orden FEATURE_ORDER.
best_tabular_model = best_model          # <<< AJUSTA al nombre de tu variable
tabular_scaler = scaler                  # <<< AJUSTA (o None si tu modelo no escala)

joblib.dump(best_tabular_model, "models/tabular_model.joblib")
print("[OK] Guardado models/tabular_model.joblib")

if tabular_scaler is not None:
    joblib.dump(tabular_scaler, "models/tabular_scaler.joblib")
    print("[OK] Guardado models/tabular_scaler.joblib")
else:
    print("[INFO] Sin scaler (modelo basado en árboles, no requiere escalado)")

# Verificación del orden de features (CRÍTICO: debe coincidir con app.py)
FEATURE_ORDER = [
    'joint_pain', 'gender', 'age', 'menopause_age', 'height__meter',
    'weight_kg_', 'smoker', 'diabetic', 'hypothyroidism',
    'number_of_pregnancies', 'seizer_disorder', 'estrogen_use', 'dialysis',
    'family_history_of_osteoporosis', 'maximum_walking_distance_km', 'bmi_',
    'obesity', 'meno_registrada', 'fractura_previa', 'occupation_cod',
    'dieta_restrictiva', 'antecedente_medico'
]
if hasattr(best_tabular_model, "n_features_in_"):
    n = best_tabular_model.n_features_in_
    assert n == len(FEATURE_ORDER), (
        f"[ERROR] Tu modelo espera {n} features pero FEATURE_ORDER tiene "
        f"{len(FEATURE_ORDER)}. Revisa que entrenaste con las mismas 22.")
    print(f"[OK] El modelo tabular usa {n} features (coincide).")

# ---------------------------------------------------------------- CNN
# <<< AJUSTA: tu mejor CNN entrenada (ej. InceptionV3+CLAHE).
try:
    best_cnn = model                     # <<< AJUSTA al nombre de tu variable
    best_cnn.save("models/cnn_model.keras")
    print("[OK] Guardado models/cnn_model.keras")
    print("[NOTA] Si tu CNN es InceptionV3, app.py la redimensiona a 299x299;")
    print("       el resto a 224x224. Verifica que coincida con tu entrenamiento.")
except NameError:
    print("[INFO] No se encontró variable 'model' para la CNN. Guárdala manualmente:")
    print("       tu_cnn.save('models/cnn_model.keras')")

# ---------------------------------------------------------------- DESCARGA
print("\nListo. Para descargar la carpeta desde Colab:")
print("  !zip -r models.zip models")
print("  from google.colab import files; files.download('models.zip')")
