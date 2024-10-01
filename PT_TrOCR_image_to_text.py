from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Charger le modèle TrOCR pré-entrainé
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Charger l'image à partir du fichier
image_path = "path/to/your/image.png"
image = Image.open(image_path).convert("RGB")

# Prétraiter l'image et effectuer l'OCR
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Texte extrait : {extracted_text}")
