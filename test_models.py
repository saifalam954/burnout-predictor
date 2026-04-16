import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("TESTING MODEL LOADING")
print("=" * 50)

try:
    print("\n1. Loading XGBoost model...")
    model = joblib.load('models/burnout_model.pkl')
    print("   ✓ Model loaded successfully")
    
    print("\n2. Loading preprocessor...")
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("   ✓ Preprocessor loaded successfully")
    
    print("\n3. Loading label encoder...")
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("   ✓ Label encoder loaded successfully")
    
    print("\n" + "=" * 50)
    print("ALL MODELS LOADED SUCCESSFULLY!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n✗ Error: {e}")