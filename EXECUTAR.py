
import sys
import os

print("="*70)
print("SISTEMA DETEÇÃO DE PLACAS BRASIL")
print("="*70)

print("\n🔍 Verificando dependências...")

faltando = []

try:
    import cv2
    print("✅ OpenCV")
except:
    print("❌ OpenCV")
    faltando.append("opencv-python")

try:
    import numpy
    print("✅ NumPy")
except:
    print("❌ NumPy")
    faltando.append("numpy")

try:
    import PIL
    print("✅ Pillow")
except:
    print("❌ Pillow")
    faltando.append("Pillow")

try:
    import pytesseract
    print("✅ Tesseract OCR")
except:
    print("⚠️  Tesseract OCR não disponível ")

try:
    import easyocr
    print("✅ EasyOCR ")
except:
    print("⚠️  EasyOCR não disponível ")

if faltando:
    print(f"\n❌ DEPENDÊNCIAS OBRIGATÓRIAS FALTANDO:")
    print(f"   {', '.join(faltando)}")
    print(f"\nINSTALE COM:")
    print(f"   pip install {' '.join(faltando)}")
    input("\nPressione ENTER para fechar...")
    sys.exit(1)

# Verificar arquivo
if not os.path.exists('sistema_placas_final.py'):
    print("\n❌ Arquivo sistema_placas_final.py não encontrado!")
    input("\nPressione ENTER para fechar...")
    sys.exit(1)


try:
    from sistema_placas_final import main
    main()
except KeyboardInterrupt:
    print("\n\n👋 Sistema encerrado pelo usuário")
except Exception as e:
    print(f"\n\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    input("\nPressione ENTER para fechar...")

