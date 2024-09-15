from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random

if not os.path.exists('./output'):
    os.makedirs('./output')

# Función para cargar imágenes por lotes
def cargar_imagenes_por_lotes(ruta_imagenes, size=(256, 256), batch_size=32):
    archivos_imagenes = [f for f in os.listdir(ruta_imagenes) if f.endswith(".jpg")]
    num_imagenes = len(archivos_imagenes)
    
    for inicio in range(0, num_imagenes, batch_size):
        imagenes_batch = [
            np.array(Image.open(os.path.join(ruta_imagenes, archivo)).convert("RGB").resize(size)) / 255.0
            for archivo in archivos_imagenes[inicio:inicio + batch_size]
        ]
        yield np.array(imagenes_batch)

# Cargar el modelo entrenado
model = load_model(r'.\data\model30_256.h5', compile=False)

# Definir la ruta de las imágenes de prueba
ruta_imagenes_prueba = r".\data\testImages"

# Definir el tamaño del batch
batch_size = 5  #

#  Mostrar predicciones
def mostrar_predicciones_aleatorias(X, predicciones, num_imagenes=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_imagenes):
        plt.subplot(3, num_imagenes, i + 1)
        plt.imshow(X[i])
        plt.title("Imagen Original")
        plt.axis('off')

        plt.subplot(3, num_imagenes, num_imagenes + i + 1)
        plt.imshow(predicciones[i].squeeze(), cmap='gray')
        plt.title("Predicción")
        plt.axis('off')
    plt.show()

# FProcesar los lotes
def procesar_por_lotes():
    lotes = cargar_imagenes_por_lotes(ruta_imagenes_prueba, batch_size=batch_size)
    for idx, X_test_batch in enumerate(lotes):
        predicciones = model.predict(X_test_batch)
        predicciones_binarizadas = (predicciones > 0.5).astype(np.uint8)

        print(f'Mostrando lote {idx + 1}')
        mostrar_predicciones_aleatorias(X_test_batch, predicciones_binarizadas, num_imagenes=len(X_test_batch))

        siguiente = input("Presiona Enter para ver el siguiente lote o 'q' para salir: ")
        if siguiente.lower() == 'q':
            print("Fin de la visualización.")
            break

# Procesar y mostrar las predicciones por lotes
procesar_por_lotes()
