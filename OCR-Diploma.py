import cv2
import pytesseract
import numpy as np
import re
from pdf2image import convert_from_path
from difflib import get_close_matches
import os
from typing import List, Tuple

TESSERACT_PATH = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
POPPLER_PATH = 'C:\\Users\\Ilya Pozhidaev\\Desktop\\Учеба\\газпром\\poppler-24.08.0\\Library\\bin'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def correct_discipline_name(raw_name: str) -> str:
    # 1. Базовая очистка
    clean = re.sub(r'[^А-Яа-яЁёA-Za-z\s-]', '', raw_name).lower().strip()
    unwanted = ["зе", " з", " е", "зс", "ззе"]
    for fragment in unwanted:
        clean = clean.replace(fragment, "")
    
    return clean

def pdf_to_images(pdf_path: str) -> List[np.ndarray]:
    """Конвертирует PDF в изображения"""
    if not os.path.exists(POPPLER_PATH):
        raise FileNotFoundError("Укажите правильный путь к Poppler в переменной POPPLER_PATH")
    
    return [np.array(img) for img in convert_from_path(pdf_path, poppler_path=POPPLER_PATH)]

def extract_grades_from_image(img: np.ndarray) -> List[Tuple[str, str]]:
    """Извлекает данные из одного изображения"""
    # Предобработка
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Распознавание
    text = pytesseract.image_to_string(thresh, lang='rus+eng', config='--oem 3 --psm 6')
    
    # Парсинг
    pattern = re.compile(
        r'(?P<discipline>.+?)\s*'
        r'(?:\d+\s*(?:з\.е\.|часов?)\s*)?'
        r'(?P<grade>зачтено|отлично|хорошо|удовлетворительно)',
        re.IGNORECASE
    )
    
    results = []
    for line in text.split('\n'):
        for match in pattern.finditer(line):
            disc = correct_discipline_name(match.group('discipline'))
            grade = match.group('grade').lower()
            results.append((disc, grade))
    
    return results

def process_diploma(pdf_path: str) -> List[Tuple[str, str]]:
    """Основная функция обработки диплома"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")
    
    images = pdf_to_images(pdf_path)
    all_results = []
    
    for img in images:
        all_results.extend(extract_grades_from_image(img))
    
    # Удаляем дубликаты (если одна дисциплина на нескольких страницах)
    return list(dict.fromkeys(all_results))

# Пример использования
if __name__ == "__main__":
    pdf_file = "chekalkina_1_1_1_3.pdf"  # Ваш PDF файл
    
    try:
        print(f"Обработка диплома: {pdf_file}")
        grades = process_diploma(pdf_file)
        
        print("\nРезультаты:")
        for i, (disc, grade) in enumerate(grades, 1):
            print(f"{i}. {disc.capitalize():<50} — {grade}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")