# Sistema de Reconhecimento de Placas Mercosul - Versão Final
# Detecta placas brasileiras (Mercosul e antigas) usando múltiplas técnicas de visão computacional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class SistemaReconhecimentoPlacasMelhorado:
    """Sistema de detecção agressiva de placas veiculares"""

    def __init__(self):
        print("🚀 Inicializando Sistema AGRESSIVO V2.0...")

        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['pt', 'en'], gpu=False, verbose=False)
                print("✅ EasyOCR configurado")
            except:
                self.easyocr_reader = None

        # Configurações otimizadas para detecção agressiva
        self.config = {
            'gaussian_kernel': (3, 3),
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'canny_low': 20,
            'canny_high': 120,
            'morph_kernel_size': (3, 3),
            'morph_rect_kernel': (5, 2),
            'placa_aspect_ratio_min': 1.8,
            'placa_aspect_ratio_max': 7.0,
            'placa_area_min': 800,
            'placa_area_max': 80000,
            'placa_width_min': 40,
            'placa_height_min': 10,
            'placa_width_max': 600,
            'placa_height_max': 200,
            'tesseract_config': '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'densidade_texto_min': 0.1,
            'densidade_texto_max': 0.9,
            'picos_minimos': 2,
        }

        print("✅ Sistema AGRESSIVO pronto!")

    def filtrar_regiao_interesse(self, imagem):
        """Utiliza imagem completa sem filtro de região"""
        h, w = imagem.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return imagem.copy(), mask

    def preprocessar_para_placas(self, imagem):
        """Aplicar múltiplos filtros de pré-processamento"""
        resultados = {'original': imagem.copy()}

        suavizada = cv2.bilateralFilter(imagem, 11, 75, 75)
        resultados['suavizada'] = suavizada

        gray = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)

        _, bin_otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resultados['bin_otsu'] = bin_otsu

        bin_adaptiva = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        resultados['bin_adaptiva'] = bin_adaptiva

        bin_adaptiva_inv = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        resultados['bin_adaptiva_inv'] = bin_adaptiva_inv

        bordas_canny = cv2.Canny(gray_clahe, 20, 120)
        resultados['bordas_canny'] = bordas_canny

        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        morph_close = cv2.morphologyEx(bin_adaptiva, cv2.MORPH_CLOSE, kernel_horizontal, iterations=2)
        resultados['morph_close_horizontal'] = morph_close

        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_opening = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_small)
        resultados['morph_opening'] = morph_opening

        return resultados

    def _tem_caracteristicas_texto(self, roi):
        if roi.size == 0:
            return False

        densidade = np.sum(roi == 255) / roi.size

        if densidade < 0.05 or densidade > 0.95:
            return False

        return True

    def _calcular_score_placa(self, roi, area, aspect_ratio):
        score = 0.5

        if 2.0 <= aspect_ratio <= 6.0:
            score += 0.3

        if 1000 <= area <= 60000:
            score += 0.2

        return min(score, 1.0)

    def _detectar_por_contornos(self, imagem_binaria, imagem_original):
        """Detectar candidatos usando contornos"""
        candidatos = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_dilatada = cv2.dilate(imagem_binaria, kernel, iterations=1)
        
        contornos, _ = cv2.findContours(img_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            if area < self.config['placa_area_min'] or area > self.config['placa_area_max']:
                continue

            x, y, w, h = cv2.boundingRect(contorno)

            if w < self.config['placa_width_min'] or w > self.config['placa_width_max']:
                continue
            if h < self.config['placa_height_min'] or h > self.config['placa_height_max']:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < self.config['placa_aspect_ratio_min'] or aspect_ratio > self.config['placa_aspect_ratio_max']:
                continue

            roi = imagem_binaria[y:y+h, x:x+w]
            
            candidatos.append({
                'bbox': (x, y, x+w, y+h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': self._calcular_score_placa(roi, area, aspect_ratio),
                'metodo': 'Contornos-Agressivo'
            })

        return candidatos

    def _detectar_por_componentes(self, imagem_binaria, imagem_original):
        """Detectar candidatos usando componentes conectados"""
        candidatos = []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagem_binaria, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < self.config['placa_area_min'] or area > self.config['placa_area_max']:
                continue

            if w < self.config['placa_width_min'] or w > self.config['placa_width_max']:
                continue
            if h < self.config['placa_height_min'] or h > self.config['placa_height_max']:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < self.config['placa_aspect_ratio_min'] or aspect_ratio > self.config['placa_aspect_ratio_max']:
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            roi = component_mask[y:y+h, x:x+w]

            candidatos.append({
                'bbox': (x, y, x+w, y+h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': self._calcular_score_placa(roi, area, aspect_ratio),
                'metodo': 'Componentes-Agressivo'
            })

        return candidatos

    def _detectar_por_bordas(self, imagem_bordas, imagem_original):
        """Detectar candidatos usando bordas Canny"""
        candidatos = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bordas_dilatadas = cv2.dilate(imagem_bordas, kernel, iterations=3)
        
        contornos, _ = cv2.findContours(bordas_dilatadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            if area < self.config['placa_area_min'] or area > self.config['placa_area_max']:
                continue

            x, y, w, h = cv2.boundingRect(contorno)

            if w < self.config['placa_width_min'] or w > self.config['placa_width_max']:
                continue
            if h < self.config['placa_height_min'] or h > self.config['placa_height_max']:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < self.config['placa_aspect_ratio_min'] or aspect_ratio > self.config['placa_aspect_ratio_max']:
                continue

            candidatos.append({
                'bbox': (x, y, x+w, y+h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': 0.6,
                'metodo': 'Bordas-Agressivo'
            })

        return candidatos

    def _filtrar_placas_candidatas(self, candidatos):
        """Remover duplicatas usando IoU e ordenar por tamanho"""
        if not candidatos:
            return []

        candidatos_unicos = []

        for candidato in candidatos:
            x1, y1, x2, y2 = candidato['bbox']

            duplicata = False
            for unico in candidatos_unicos:
                ux1, uy1, ux2, uy2 = unico['bbox']

                inter_x1, inter_y1 = max(x1, ux1), max(y1, uy1)
                inter_x2, inter_y2 = min(x2, ux2), min(y2, uy2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ux2 - ux1) * (uy2 - uy1)
                    iou = inter_area / (area1 + area2 - inter_area + 1e-5)

                    if iou > 0.3:
                        duplicata = True
                        if candidato['area'] < unico['area']:
                            candidatos_unicos.remove(unico)
                            candidatos_unicos.append(candidato)
                        break

            if not duplicata:
                candidatos_unicos.append(candidato)

        candidatos_unicos.sort(key=lambda x: x['area'])
        
        return candidatos_unicos

    def _ocr_rapido_tesseract(self, imagem):
        """OCR rápido para validação preliminar"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        resultados = []
        
        try:
            if len(imagem.shape) == 3:
                imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            
            texto1 = pytesseract.image_to_string(imagem, config='--psm 8 --oem 3')
            resultados.append(texto1)
            
            h, w = imagem.shape
            img_2x = cv2.resize(imagem, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            texto2 = pytesseract.image_to_string(img_2x, config='--psm 8 --oem 3')
            resultados.append(texto2)
            
            _, img_thresh = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            texto3 = pytesseract.image_to_string(img_thresh, config='--psm 8 --oem 3')
            resultados.append(texto3)
            
            texto_final = max(resultados, key=lambda x: len(x.strip()))
            return texto_final.strip().replace(' ', '').replace('\n', '').upper()
        except:
            return ""

    def _ocr_rapido_easyocr(self, imagem):
        """OCR rápido EasyOCR"""
        if self.easyocr_reader is None:
            return ""
        try:
            results = self.easyocr_reader.readtext(imagem, detail=0)
            return "".join(results).replace(' ', '').upper()
        except:
            return ""

    def _validar_texto_placa(self, texto1, texto2):
        """Validação de texto preliminar"""
        textos = [t for t in [texto1, texto2] if t and len(t) >= 4]
        if not textos:
            return 0.0

        melhor_score = 0.0

        for texto in textos:
            score = 0.0

            if 4 <= len(texto) <= 9:
                score += 0.3

            letras = sum(1 for c in texto if c.isalpha())
            numeros = sum(1 for c in texto if c.isdigit())

            if letras >= 1 and numeros >= 1:
                score += 0.5

            melhor_score = max(melhor_score, score)

        return min(melhor_score, 1.0)

    def _validar_com_ocr_preliminar(self, candidatos, imagem_original):
        """Validar candidatos com OCR rápido"""
        placas_validadas = []
        
        img_h, img_w = imagem_original.shape[:2]
        img_area = img_h * img_w
        
        candidatos_filtrados = []
        for candidato in candidatos:
            x1, y1, x2, y2 = candidato['bbox']
            w_cand = x2 - x1
            h_cand = y2 - y1
            area_cand = candidato['area']
            
            pct_largura = (w_cand / img_w) * 100
            pct_altura = (h_cand / img_h) * 100
            pct_area = (area_cand / img_area) * 100
            
            if pct_largura > 80 or pct_altura > 60 or pct_area > 20:
                continue
            
            candidatos_filtrados.append(candidato)
        
        for candidato in candidatos_filtrados[:10]:
            x1, y1, x2, y2 = candidato['bbox']
            
            margin = 5
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(imagem_original.shape[1], x2 + margin)
            y2 = min(imagem_original.shape[0], y2 + margin)
            
            roi = imagem_original[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            texto_tesseract = self._ocr_rapido_tesseract(roi)
            texto_easyocr = self._ocr_rapido_easyocr(roi)

            score_texto = self._validar_texto_placa(texto_easyocr, texto_tesseract)

            if score_texto > 0.2 or len(texto_tesseract) >= 5 or len(texto_easyocr) >= 5:
                candidato['confianca'] = candidato['score'] * 0.5 + score_texto * 0.5
                candidato['imagem_placa'] = roi
                candidato['texto_preliminar'] = texto_easyocr or texto_tesseract
                candidato['bbox'] = (x1, y1, x2, y2)
                placas_validadas.append(candidato)

        return placas_validadas

    def _isolar_letras_placa(self, imagem):
        """
        Isolar apenas as letras da placa
        Remove BRASIL, BR, bordas e ruído
        """
        if len(imagem.shape) == 3:
            gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagem.copy()

        h, w = gray.shape

        escala = 5
        gray_grande = cv2.resize(gray, (w*escala, h*escala), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray_grande)

        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray_clahe, -1, kernel_sharp)

        denoised = cv2.fastNlMeansDenoising(sharpened, h=10)

        _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresh_adapt = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 15, 2)
        
        thresh = thresh_otsu

        center_h = thresh.shape[0] // 2
        center_region = thresh[center_h-20:center_h+20, :]
        if np.mean(center_region) > 127:
            thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        mask_limpa = np.zeros_like(thresh)
        
        altura_min_letra = thresh.shape[0] * 0.35
        altura_max_letra = thresh.shape[0] * 0.75
        
        largura_min_letra = thresh.shape[0] * 0.15
        largura_max_letra = thresh.shape[0] * 1.2
        
        componentes_validos = []
        
        for i in range(1, num_labels):
            x, y, w_comp, h_comp, area = stats[i]
            
            margin = int(thresh.shape[0] * 0.05)
            if x < margin or y < margin:
                continue
            if x + w_comp > thresh.shape[1] - margin:
                continue
            if y + h_comp > thresh.shape[0] - margin:
                continue
            
            if h_comp < altura_min_letra or h_comp > altura_max_letra:
                continue
            
            if w_comp < largura_min_letra or w_comp > largura_max_letra:
                continue
            
            if area < 50 or area > thresh.shape[0] * thresh.shape[1] * 0.12:
                continue
            
            aspect = w_comp / float(h_comp)
            if aspect < 0.15 or aspect > 1.5:
                continue
            
            center_y = y + h_comp / 2.0
            img_center_y = thresh.shape[0] / 2.0
            distancia_centro = abs(center_y - img_center_y)
            
            if distancia_centro > thresh.shape[0] * 0.3:
                continue
            
            componentes_validos.append({
                'label': i,
                'x': x,
                'area': area,
                'height': h_comp
            })
        
        if componentes_validos:
            componentes_validos.sort(key=lambda c: c['x'])
            
            if len(componentes_validos) > 10:
                alturas = [c['height'] for c in componentes_validos]
                altura_mediana = sorted(alturas)[len(alturas)//2]
                
                componentes_validos = [
                    c for c in componentes_validos 
                    if abs(c['height'] - altura_mediana) < altura_mediana * 0.3
                ]
            
            for comp in componentes_validos:
                mask_limpa[labels == comp['label']] = 255

        if np.sum(mask_limpa) < 100:
            mask_limpa = thresh

        kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_limpa = cv2.morphologyEx(mask_limpa, cv2.MORPH_OPEN, kernel_final)
        mask_limpa = cv2.morphologyEx(mask_limpa, cv2.MORPH_CLOSE, kernel_final)

        pad = 20
        mask_limpa = cv2.copyMakeBorder(mask_limpa, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

        return mask_limpa

    def _ocr_tesseract_completo(self, imagem):
        """OCR completo com isolamento de letras"""
        if not TESSERACT_AVAILABLE:
            return ""

        resultados = []

        try:
            img_letras_isoladas = self._isolar_letras_placa(imagem)
            
            configs = [
                '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            for config in configs:
                try:
                    texto = pytesseract.image_to_string(img_letras_isoladas, config=config)
                    texto_limpo = texto.strip().replace(' ', '').replace('\n', '').upper()
                    if len(texto_limpo) >= 5:
                        resultados.append(texto_limpo)
                except:
                    pass
        except:
            pass

        try:
            if len(imagem.shape) == 3:
                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagem.copy()
            
            h, w = gray.shape
            gray_3x = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_clahe = clahe.apply(gray_3x)
            
            _, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            texto = pytesseract.image_to_string(thresh, config='--psm 8 --oem 3')
            texto_limpo = texto.strip().replace(' ', '').replace('\n', '').upper()
            if len(texto_limpo) >= 5:
                resultados.append(texto_limpo)
        except:
            pass

        try:
            if len(imagem.shape) == 3:
                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagem.copy()
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            h, w = thresh.shape
            thresh_grande = cv2.resize(thresh, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            
            texto = pytesseract.image_to_string(thresh_grande, config='--psm 8 --oem 3')
            texto_limpo = texto.strip().replace(' ', '').replace('\n', '').upper()
            if len(texto_limpo) >= 5:
                resultados.append(texto_limpo)
        except:
            pass

        if resultados:
            return max(resultados, key=len)
        
        return ""

    def _ocr_easyocr_completo(self, imagem):
        """OCR EasyOCR completo com isolamento"""
        if self.easyocr_reader is None:
            return ""

        resultados = []

        try:
            img_letras_isoladas = self._isolar_letras_placa(imagem)
            results = self.easyocr_reader.readtext(img_letras_isoladas, detail=0, paragraph=False)
            texto = "".join(results).replace(' ', '').upper()
            if len(texto) >= 5:
                resultados.append(texto)
        except:
            pass

        try:
            if len(imagem.shape) == 3:
                img_processada = imagem
            else:
                img_processada = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
            
            h, w = img_processada.shape[:2]
            img_grande = cv2.resize(img_processada, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            
            results = self.easyocr_reader.readtext(img_grande, detail=0, paragraph=False)
            texto = "".join(results).replace(' ', '').upper()
            if len(texto) >= 5:
                resultados.append(texto)
        except:
            pass

        try:
            if len(imagem.shape) == 3:
                gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagem.copy()
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            h, w = enhanced.shape
            enhanced_grande = cv2.resize(enhanced, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            
            enhanced_bgr = cv2.cvtColor(enhanced_grande, cv2.COLOR_GRAY2BGR)
            
            results = self.easyocr_reader.readtext(enhanced_bgr, detail=0, paragraph=False)
            texto = "".join(results).replace(' ', '').upper()
            if len(texto) >= 5:
                resultados.append(texto)
        except:
            pass

        if resultados:
            return max(resultados, key=len)
        
        return ""

    def _extrair_placa_do_texto(self, texto):
        """Extrair apenas os 7 caracteres da placa, removendo palavras extras"""
        import re
        
        chars_validos = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto_limpo = ''.join([c for c in texto.upper() if c in chars_validos])
        
        palavras_remover = ['BRASIL', 'BR', 'MERCOSUL', 'MERCO', 'SUL']
        
        for palavra in palavras_remover:
            texto_limpo = texto_limpo.replace(palavra, '')
        
        match_mercosul = re.findall(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', texto_limpo)
        if match_mercosul:
            return match_mercosul[0]
        
        match_antiga = re.findall(r'[A-Z]{3}[0-9]{4}', texto_limpo)
        if match_antiga:
            return match_antiga[0]
        
        for i in range(len(texto_limpo) - 6):
            chunk = texto_limpo[i:i+7]
            if len(chunk) == 7:
                if chunk[:3].isalpha():
                    num_count = sum(1 for c in chunk if c.isdigit())
                    if num_count >= 2:
                        return chunk
        
        if len(texto_limpo) == 7:
            return texto_limpo
        
        if len(texto_limpo) > 7:
            return texto_limpo[-7:]
        
        return texto_limpo

    def _pos_processar_texto(self, texto):
        """Aplicar correções inteligentes e formatação"""
        placa_extraida = self._extrair_placa_do_texto(texto)
        
        if len(placa_extraida) == 7:
            corrigido = list(placa_extraida)
            
            for i in range(3):
                if corrigido[i].isdigit():
                    mapa = {'0':'O', '6':'G', '1':'I', '5':'S', '8':'B', '2':'Z'}
                    corrigido[i] = mapa.get(corrigido[i], corrigido[i])
            
            if corrigido[3].isalpha():
                mapa = {'O':'0', 'I':'1', 'S':'5', 'G':'6', 'B':'8', 'Z':'2'}
                corrigido[3] = mapa.get(corrigido[3], corrigido[3])
            
            for i in range(5, 7):
                if corrigido[i].isalpha():
                    mapa = {'O':'0', 'I':'1', 'S':'5', 'G':'6', 'B':'8', 'Z':'2'}
                    corrigido[i] = mapa.get(corrigido[i], corrigido[i])
            
            texto_corrigido = ''.join(corrigido)
            return f"{texto_corrigido[:3]}-{texto_corrigido[3:]}"

        return placa_extraida

    def detectar_placas_melhorado(self, imagem):
        """Detecção com todas as estratégias disponíveis"""
        try:
            imagem_roi, _ = self.filtrar_regiao_interesse(imagem)
            prep_results = self.preprocessar_para_placas(imagem_roi)

            candidatos = []
            
            try:
                c1 = self._detectar_por_contornos(prep_results['morph_opening'], imagem)
                candidatos.extend(c1)
            except Exception as e:
                print(f"Erro contornos morph_opening: {e}")
            
            try:
                c2 = self._detectar_por_contornos(prep_results['bin_adaptiva'], imagem)
                candidatos.extend(c2)
            except Exception as e:
                print(f"Erro contornos bin_adaptiva: {e}")
            
            try:
                c3 = self._detectar_por_contornos(prep_results['bin_otsu'], imagem)
                candidatos.extend(c3)
            except Exception as e:
                print(f"Erro contornos bin_otsu: {e}")
            
            try:
                c4 = self._detectar_por_componentes(prep_results['bin_adaptiva'], imagem)
                candidatos.extend(c4)
            except Exception as e:
                print(f"Erro componentes bin_adaptiva: {e}")
            
            try:
                c5 = self._detectar_por_componentes(prep_results['morph_opening'], imagem)
                candidatos.extend(c5)
            except Exception as e:
                print(f"Erro componentes morph_opening: {e}")
            
            try:
                c6 = self._detectar_por_bordas(prep_results['bordas_canny'], imagem)
                candidatos.extend(c6)
            except Exception as e:
                print(f"Erro bordas canny: {e}")

            candidatos_filtrados = self._filtrar_placas_candidatas(candidatos)
            
            if not candidatos_filtrados:
                print("⚠️ Nenhum candidato detectado! Criando candidato fallback com imagem inteira.")
                h, w = imagem.shape[:2]
                candidatos_filtrados = [{
                    'bbox': (0, 0, w, h),
                    'area': w * h,
                    'aspect_ratio': w / h,
                    'score': 0.1,
                    'metodo': 'Fallback-ImagemInteira'
                }]
            
            placas_validadas = self._validar_com_ocr_preliminar(candidatos_filtrados, imagem)

            return placas_validadas[:5]
        
        except Exception as e:
            print(f"❌ ERRO CRÍTICO em detectar_placas_melhorado: {e}")
            import traceback
            traceback.print_exc()
            h, w = imagem.shape[:2]
            return [{
                'bbox': (0, 0, w, h),
                'area': w * h,
                'aspect_ratio': w / h,
                'score': 0.1,
                'metodo': 'Fallback-Erro',
                'imagem_placa': imagem
            }]

    def _validar_placa_final(self, texto, score_deteccao):
        """Validar se texto corresponde a uma placa válida"""
        import re
        
        if not texto:
            return False, 0.0
        
        texto_limpo = texto.replace('-', '')
        
        if len(texto_limpo) < 6 or len(texto_limpo) > 8:
            return False, 0.0
        
        confianca = 0.0
        
        if len(texto_limpo) == 7:
            if re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', texto_limpo):
                confianca = 1.0
            
            elif re.match(r'^[A-Z]{3}[0-9]{4}$', texto_limpo):
                confianca = 0.9
            
            elif re.match(r'^[A-Z]{2,3}', texto_limpo):
                letras = sum(1 for c in texto_limpo if c.isalpha())
                numeros = sum(1 for c in texto_limpo if c.isdigit())
                
                if letras >= 2 and numeros >= 3:
                    confianca = 0.7
                elif letras >= 2 and numeros >= 2:
                    confianca = 0.5
                else:
                    return False, 0.0
            else:
                letras = sum(1 for c in texto_limpo if c.isalpha())
                numeros = sum(1 for c in texto_limpo if c.isdigit())
                
                if letras >= 2 and numeros >= 2:
                    confianca = 0.4
                else:
                    return False, 0.0
        
        else:
            letras = sum(1 for c in texto_limpo if c.isalpha())
            numeros = sum(1 for c in texto_limpo if c.isdigit())
            
            if letras >= 2 and numeros >= 2:
                confianca = 0.5
            else:
                return False, 0.0
        
        confianca_final = confianca * 0.6 + score_deteccao * 0.4
        
        return confianca_final >= 0.5, confianca_final

    def processar_imagem(self, caminho_imagem, log_callback=None):
        """Processar imagem completa com log detalhado"""
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                log(f"❌ ERRO: Não foi possível carregar: {caminho_imagem}")
                return {'erro': f'Não foi possível carregar: {caminho_imagem}'}

            nome_arquivo = os.path.basename(caminho_imagem)
            log(f"📸 Imagem carregada: {nome_arquivo}")
            log(f"📐 Dimensões: {imagem.shape[1]}x{imagem.shape[0]} pixels")

            resultado = {
                'caminho': caminho_imagem,
                'nome_arquivo': nome_arquivo,
                'imagem_original': imagem,
                'placas_detectadas': [],
                'resultados_ocr': []
            }

            log("\n🔍 Iniciando detecção de candidatos...")
            log("🔬 Aplicando pré-processamento:")
            log("   1️⃣ Filtro de Região de Interesse")
            log("   2️⃣ Escala de Cinza")
            log("   3️⃣ CLAHE (Contraste)")
            log("   4️⃣ Binarização Otsu")
            log("   5️⃣ Binarização Adaptativa")
            log("   6️⃣ Detecção de Bordas Canny")
            log("   7️⃣ Morfologia (Close + Open)")
            
            placas = self.detectar_placas_melhorado(imagem)
            resultado['placas_detectadas'] = placas
            
            log(f"\n📦 {len(placas)} candidato(s) encontrado(s) após todos os filtros")
            
            if placas:
                log(f"\n📊 Candidatos ordenados por tamanho (MENOR = MELHOR):")
                for idx, p in enumerate(placas[:5], 1):
                    w = p['bbox'][2] - p['bbox'][0]
                    h = p['bbox'][3] - p['bbox'][1]
                    area = p['area']
                    metodo = p.get('metodo', 'N/A')
                    pct_img = (area / (imagem.shape[0] * imagem.shape[1])) * 100
                    log(f"   {idx}. {w}x{h} (área: {area}, {pct_img:.1f}% img) - {metodo}")
                log("")
            else:
                log("⚠️ ATENÇÃO: Nenhum candidato passou nos filtros!")
                log("   Sistema vai tentar OCR em regiões alternativas...")

            placa_valida_encontrada = False
        
        except Exception as e:
            log(f"❌ ERRO CRÍTICO no carregamento: {e}")
            import traceback
            traceback.print_exc()
            return {'erro': f'Erro crítico: {e}'}

        for i, placa in enumerate(placas):
            if placa_valida_encontrada:
                log(f"\n⏭️  Ignorando candidato {i+1} (placa válida já encontrada)")
                continue
            
            log(f"\n🎯 Processando candidato {i+1}/{len(placas)}...")
            
            try:
                imagem_placa = placa.get('imagem_placa')
                if imagem_placa is None:
                    log(f"   ⚠️ Imagem da placa não disponível, pulando...")
                    continue
                
                log(f"   📐 Dimensões: {imagem_placa.shape[1]}x{imagem_placa.shape[0]}")
                log(f"   🔧 Método detecção: {placa.get('metodo', 'N/A')}")
                
                log(f"\n   🔬 Aplicando tratamentos OCR:")
                log(f"      • Isolamento de letras (removendo BRASIL, BR, bordas)")
                log(f"      • Ampliação 5x para maior resolução")
                log(f"      • CLAHE para contraste")
                log(f"      • Sharpening para nitidez")
                log(f"      • Múltiplas binarizações")
                log(f"      • Denoising (remoção de ruído)")
                log(f"      • Morfologia para conectar letras")
                
                log(f"\n   📖 Executando OCR Tesseract...")
                texto_tesseract = self._ocr_tesseract_completo(imagem_placa)
                log(f"   📝 Tesseract bruto: '{texto_tesseract}'")
                
                log(f"   📖 Executando OCR EasyOCR...")
                texto_easyocr = self._ocr_easyocr_completo(imagem_placa)
                log(f"   📝 EasyOCR bruto: '{texto_easyocr}'")

                log(f"\n   🔧 Aplicando pós-processamento:")
                log(f"      • Extração de placa (7 caracteres)")
                log(f"      • Remoção de palavras (BRASIL, BR, MERCOSUL)")
                log(f"      • Correções inteligentes (G↔6, O↔0, etc)")
                log(f"      • Formatação final")
                
                final_tesseract = self._pos_processar_texto(texto_tesseract)
                final_easyocr = self._pos_processar_texto(texto_easyocr)
                
                log(f"   ✅ Tesseract final: '{final_tesseract}'")
                log(f"   ✅ EasyOCR final: '{final_easyocr}'")
                
                melhor_texto = final_easyocr if final_easyocr else final_tesseract
                score_deteccao = placa.get('score', 0)
            
            except Exception as e:
                log(f"   ❌ Erro ao processar candidato: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            valida, confianca_final = self._validar_placa_final(melhor_texto, score_deteccao)
            
            if valida:
                log(f"   ✅ PLACA VÁLIDA! Confiança: {confianca_final:.1%}")
                placa_valida_encontrada = True
            else:
                log(f"   ⚠️  Não parece ser placa válida (confiança: {confianca_final:.1%})")
                continue

            resultado_ocr = {
                'placa_id': i,
                'bbox': placa['bbox'],
                'confianca_deteccao': confianca_final,
                'metodo_deteccao': placa.get('metodo', 'N/A'),
                'score_qualidade': placa.get('score', 0),
                'dimensoes': f"{placa['bbox'][2] - placa['bbox'][0]}x{placa['bbox'][3] - placa['bbox'][1]}",
                'aspect_ratio': placa.get('aspect_ratio', 0),
                'area': placa.get('area', 0),
                'imagem_placa': imagem_placa,
                'tesseract': {
                    'texto_bruto': texto_tesseract,
                    'texto_final': final_tesseract
                },
                'easyocr': {
                    'texto_bruto': texto_easyocr,
                    'texto_final': final_easyocr
                },
                'placa_valida': valida
            }

            resultado['resultados_ocr'].append(resultado_ocr)
            
            if placa_valida_encontrada:
                log(f"   🎯 Placa encontrada! Parando processamento.")
                break

        return resultado


class PainelPlacasMercosulFinal:
    """Interface gráfica para o sistema de reconhecimento"""

    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Sistema Melhorado V2.0 - Placas Mercosul Brasil")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        self.sistema = None
        self.imagem_atual = None
        self.caminho_imagem = None

        self.configurar_interface()
        self.inicializar_sistema()

    def configurar_interface(self):
        """Configurar layout da interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(2, weight=2)
        main_frame.rowconfigure(1, weight=1)

        titulo = ttk.Label(main_frame, text="🚗 Sistema MELHORADO V2.0 - Placas Mercosul Brasil",
                          font=('Arial', 16, 'bold'), foreground='darkgreen')
        titulo.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        frame_esquerdo = ttk.Frame(main_frame)
        frame_esquerdo.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        frame_esquerdo.rowconfigure(1, weight=1)

        frame_controles = ttk.LabelFrame(frame_esquerdo, text="📁 Controles", padding="10")
        frame_controles.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.btn_carregar = ttk.Button(frame_controles, text="📷 Carregar Imagem",
                                       command=self.carregar_imagem, width=25)
        self.btn_carregar.grid(row=0, column=0, pady=3, sticky=tk.W+tk.E)

        self.btn_processar = ttk.Button(frame_controles, text="🔍 Processar",
                                        command=self.processar_imagem, width=25, state='disabled')
        self.btn_processar.grid(row=1, column=0, pady=3, sticky=tk.W+tk.E)

        self.progress = ttk.Progressbar(frame_controles, mode='indeterminate')
        self.progress.grid(row=2, column=0, pady=3, sticky=tk.W+tk.E)

        self.label_status = ttk.Label(frame_controles, text="🔄 Iniciando...", foreground='orange')
        self.label_status.grid(row=3, column=0, pady=5, sticky=tk.W)

        frame_stats = ttk.LabelFrame(frame_controles, text="📊 Estatísticas", padding="10")
        frame_stats.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 10))

        self.label_placas = ttk.Label(frame_stats, text="Placas: 0")
        self.label_placas.grid(row=0, column=0, sticky=tk.W)

        self.label_metodo = ttk.Label(frame_stats, text="Método:")
        self.label_metodo.grid(row=1, column=0, sticky=tk.W)

        frame_resultado = ttk.LabelFrame(frame_controles, text="🎯 PLACA", padding="10")
        frame_resultado.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 0))

        self.label_placa = ttk.Label(frame_resultado, text="---",
                                     font=('Arial', 18, 'bold'), foreground='green')
        self.label_placa.grid(row=0, column=0, sticky=tk.W)

        self.label_tipo = ttk.Label(frame_resultado, text="", font=('Arial', 9))
        self.label_tipo.grid(row=1, column=0, sticky=tk.W)

        self.label_confianca = ttk.Label(frame_resultado, text="")
        self.label_confianca.grid(row=2, column=0, sticky=tk.W)

        frame_etapas = ttk.LabelFrame(frame_esquerdo, text="🔬 Etapas Visuais", padding="10")
        frame_etapas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        frame_etapas.columnconfigure(0, weight=1)
        frame_etapas.rowconfigure(0, weight=1)

        self.canvas_etapas = tk.Canvas(frame_etapas, bg='white', width=280, height=400)
        self.canvas_etapas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scroll_etapas = ttk.Scrollbar(frame_etapas, orient="vertical", command=self.canvas_etapas.yview)
        scroll_etapas.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas_etapas.configure(yscrollcommand=scroll_etapas.set)

        frame_visualizacao = ttk.LabelFrame(main_frame, text="🖼️ RESULTADO", padding="10")
        frame_visualizacao.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        frame_visualizacao.columnconfigure(0, weight=1)
        frame_visualizacao.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame_visualizacao, bg='white', width=600, height=500)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        v_scrollbar = ttk.Scrollbar(frame_visualizacao, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(frame_visualizacao, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)

        frame_log = ttk.LabelFrame(main_frame, text="📋 Log Detalhado", padding="10")
        frame_log.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        frame_log.columnconfigure(0, weight=1)
        frame_log.rowconfigure(0, weight=1)

        self.text_log = scrolledtext.ScrolledText(frame_log, width=50, height=30,
                                                  font=('Consolas', 9), wrap=tk.WORD)
        self.text_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        btn_limpar = ttk.Button(frame_log, text="🗑️ Limpar Log",
                                command=self.limpar_log, width=15)
        btn_limpar.grid(row=1, column=0, pady=5)

        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        self.etapas_imagens = []

    def inicializar_sistema(self):
        """Inicializar sistema em thread separada"""
        def init():
            try:
                self.adicionar_log("🚀 Inicializando Sistema AGRESSIVO V2.0...")
                self.sistema = SistemaReconhecimentoPlacasMelhorado()
                self.label_status.config(text="✅ Sistema Pronto", foreground='green')
                self.adicionar_log("✅ Sistema AGRESSIVO pronto!")
                self.adicionar_log("🎯 Filtros relaxados para detectar mais placas")
            except Exception as e:
                self.label_status.config(text=f"❌ Erro: {e}", foreground='red')
                self.adicionar_log(f"❌ Erro: {e}")

        thread = threading.Thread(target=init)
        thread.daemon = True
        thread.start()

    def carregar_imagem(self):
        """Carregar imagem do disco"""
        caminho = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
        )

        if caminho:
            try:
                self.caminho_imagem = caminho
                self.imagem_atual = cv2.imread(caminho)

                if self.imagem_atual is None:
                    messagebox.showerror("Erro", "Não foi possível carregar a imagem!")
                    return

                nome_arquivo = os.path.basename(caminho)
                h, w = self.imagem_atual.shape[:2]

                self.mostrar_imagem_canvas(self.imagem_atual)
                self.btn_processar.config(state='normal')

                self.adicionar_log(f"\n📸 Imagem: {nome_arquivo}")
                self.adicionar_log(f"📐 {w}x{h} pixels")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro: {str(e)}")

    def mostrar_imagem_canvas(self, imagem):
        """Exibir imagem no canvas"""
        if len(imagem.shape) == 3:
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        else:
            imagem_rgb = imagem

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            h, w = imagem_rgb.shape[:2]

            scale = min(canvas_width / w, canvas_height / h, 1.0)

            new_w = int(w * scale)
            new_h = int(h * scale)

            imagem_resized = cv2.resize(imagem_rgb, (new_w, new_h))

            pil_image = Image.fromarray(imagem_resized)
            self.photo = ImageTk.PhotoImage(pil_image)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Redimensionar canvas"""
        if hasattr(self, 'imagem_atual') and self.imagem_atual is not None:
            self.mostrar_imagem_canvas(self.imagem_atual)

    def processar_imagem(self):
        """Processar imagem em thread separada"""
        if self.imagem_atual is None or self.sistema is None:
            messagebox.showerror("Erro", "Carregue uma imagem!")
            return

        def processar():
            try:
                self.btn_processar.config(state='disabled')
                self.progress.start()
                self.label_status.config(text="🔄 Processando...", foreground='orange')

                self.adicionar_log("\n" + "="*60)
                self.adicionar_log("🚀 PROCESSANDO - MODO AGRESSIVO")
                self.adicionar_log("🔬 COM ISOLAMENTO DE LETRAS")
                self.adicionar_log("="*60)

                resultado = self.sistema.processar_imagem(self.caminho_imagem, log_callback=self.adicionar_log)

                if 'erro' in resultado:
                    self.adicionar_log(f"❌ {resultado['erro']}")
                    self.mostrar_resultado_final(None)
                else:
                    self.mostrar_resultado_final(resultado)
                    self.mostrar_etapas_processamento(resultado)

            except Exception as e:
                self.adicionar_log(f"❌ ERRO: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                self.btn_processar.config(state='normal')
                self.progress.stop()
                self.label_status.config(text="✅ Pronto", foreground='green')

        thread = threading.Thread(target=processar)
        thread.daemon = True
        thread.start()

    def mostrar_resultado_final(self, resultado):
        """Exibir resultado final com placa detectada"""
        self.adicionar_log("\n" + "="*60)
        self.adicionar_log("🏁 RESULTADO FINAL")
        self.adicionar_log("="*60)

        if resultado is None or not resultado.get('resultados_ocr'):
            self.label_placa.config(text="❌ Não detectada", foreground='red')
            self.label_tipo.config(text="")
            self.label_confianca.config(text="")
            self.label_placas.config(text="Placas: 0")
            self.label_metodo.config(text="")
            self.adicionar_log("❌ Nenhuma placa válida detectada")
            return

        ocr_result = resultado['resultados_ocr'][0]
        
        if not ocr_result.get('placa_valida', False):
            self.label_placa.config(text="❌ Não validada", foreground='red')
            self.label_tipo.config(text="Candidatos encontrados mas não validados")
            self.label_confianca.config(text="")
            self.label_placas.config(text="Placas: 0")
            self.label_metodo.config(text="")
            self.adicionar_log("⚠️ Candidatos encontrados mas nenhum passou na validação")
            return
        
        texto = ocr_result['easyocr']['texto_final'] or ocr_result['tesseract']['texto_final']
        confianca = ocr_result['confianca_deteccao']
        metodo = ocr_result['metodo_deteccao']
        
        import re
        texto_sem_hifen = texto.replace('-', '')
        if len(texto_sem_hifen) == 7:
            if re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', texto_sem_hifen):
                tipo = "✅ Mercosul (ABC1D23)"
            elif re.match(r'^[A-Z]{3}[0-9]{4}$', texto_sem_hifen):
                tipo = "✅ Antiga (ABC1234)"
            else:
                tipo = "Formato detectado"
        else:
            tipo = "Texto detectado"

        self.label_placa.config(text=texto if texto else "---", foreground='green')
        self.label_tipo.config(text=tipo)
        self.label_confianca.config(text=f"Confiança: {confianca:.1%}")
        self.label_placas.config(text="Placas válidas: 1")
        self.label_metodo.config(text=f"Método: {metodo}")

        self.adicionar_log(f"✅ PLACA VÁLIDA: {texto}")
        self.adicionar_log(f"📋 Tipo: {tipo}")
        self.adicionar_log(f"📊 Confiança: {confianca:.1%}")
        self.adicionar_log(f"🔧 Método: {metodo}")
        self.adicionar_log(f"📐 Dimensões: {ocr_result['dimensoes']}")
        
        self.adicionar_log(f"\n🔍 Debug OCR:")
        self.adicionar_log(f"   Tesseract bruto: '{ocr_result['tesseract']['texto_bruto']}'")
        self.adicionar_log(f"   EasyOCR bruto: '{ocr_result['easyocr']['texto_bruto']}'")

        try:
            img_resultado = resultado['imagem_original'].copy()
            x1, y1, x2, y2 = ocr_result['bbox']

            cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (0, 255, 0), 5)

            if texto:
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(img_resultado, (x1, y1-50), (x1+tw+20, y1), (0, 255, 0), -1)
                cv2.putText(img_resultado, texto, (x1+10, y1-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            cv2.putText(img_resultado, metodo, (x1, y2+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(img_resultado, f"Conf: {confianca:.1%}", (x1, y2+60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.mostrar_imagem_canvas(img_resultado)
        except Exception as e:
            self.adicionar_log(f"⚠️ Erro ao desenhar resultado: {e}")

    def adicionar_log(self, texto):
        """Adicionar texto ao log"""
        self.text_log.insert(tk.END, texto + "\n")
        self.text_log.see(tk.END)
        self.root.update_idletasks()

    def limpar_log(self):
        """Limpar log"""
        self.text_log.delete(1.0, tk.END)
    
    def mostrar_etapas_processamento(self, resultado):
        """Mostrar etapas visuais do processamento"""
        try:
            if not resultado or 'resultados_ocr' not in resultado or not resultado['resultados_ocr']:
                self.adicionar_log("⚠️ Sem resultados para mostrar etapas")
                return
            
            self.canvas_etapas.delete("all")
            self.etapas_imagens = []
            
            self.adicionar_log("\n🔬 Gerando visualização das etapas...")
            
            ocr = resultado['resultados_ocr'][0]
            img_placa = ocr.get('imagem_placa')
            
            if img_placa is None:
                self.adicionar_log("⚠️ Imagem da placa não disponível")
                return
            
            etapas = []
            
            etapas.append(("1. Placa Recortada", img_placa))
            
            if len(img_placa.shape) == 3:
                gray = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_placa.copy()
            etapas.append(("2. Escala Cinza", gray))
            
            h, w = gray.shape
            gray_5x = cv2.resize(gray, (w*5, h*5), interpolation=cv2.INTER_CUBIC)
            etapas.append(("3. Ampliada 5x", gray_5x))
            
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(gray_5x)
            etapas.append(("4. CLAHE (Contraste)", img_clahe))
            
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(img_clahe, -1, kernel_sharp)
            etapas.append(("5. Sharpening", sharpened))
            
            _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            etapas.append(("6. Binarizada", thresh))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            etapas.append(("7. Morfologia Final", morph))
            
            y_offset = 10
            
            for nome, img in etapas:
                h, w = img.shape[:2] if len(img.shape) == 2 else img.shape[:2]
                max_width = 270
                scale = min(max_width / w, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    if len(img_resized.shape) == 2:
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    
                    pil_img = Image.fromarray(img_rgb)
                    photo = ImageTk.PhotoImage(pil_img)
                    self.etapas_imagens.append(photo)
                    
                    self.canvas_etapas.create_text(10, y_offset, text=nome, anchor=tk.NW, 
                                                  font=('Arial', 9, 'bold'), fill='darkgreen')
                    self.canvas_etapas.create_image(10, y_offset + 20, anchor=tk.NW, image=photo)
                    
                    y_offset += new_h + 35
            
            self.canvas_etapas.configure(scrollregion=self.canvas_etapas.bbox("all"))
            self.adicionar_log("✅ Etapas visuais geradas")
            
        except Exception as e:
            self.adicionar_log(f"❌ Erro ao gerar etapas: {e}")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = PainelPlacasMercosulFinal(root)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    print("🚗 Sistema AGRESSIVO V2.0 - Detecta até placas difíceis")
    print("="*70)
    main()
