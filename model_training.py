import random
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

# Definir la arquitectura del modelo UNet con tamaño de entrada 256x256
def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(32, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(64, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(128, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(512, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3], axis=-1)
    c5 = Conv2D(256, (3, 3), padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(256, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2], axis=-1)
    c6 = Conv2D(128, (3, 3), padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(128, (3, 3), padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1], axis=-1)
    c7 = Conv2D(64, (3, 3), padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(64, (3, 3), padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    # Compile model
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss=BinaryCrossentropy(), 
                  metrics=[BinaryAccuracy()])
    
    return model

def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(256, 256)):
    imagenes = []
    mascaras = []
    for archivo in os.listdir(ruta_imagenes):
        if archivo.endswith(".jpg"):
            mascara_path = os.path.join(ruta_mascaras, archivo)
            if not os.path.exists(mascara_path):
                print(f"Advertencia: No se encontró la máscara correspondiente para {archivo}. Se omite este archivo.")
                continue
            imagen_path = os.path.join(ruta_imagenes, archivo)
            imagen = Image.open(imagen_path).convert("RGB")
            imagen = imagen.resize(tamaño)
            imagen_array = np.array(imagen) / 255.0
            imagenes.append(imagen_array)
            
            mascara = Image.open(mascara_path).convert("L")
            mascara = mascara.resize(tamaño)
            mascara_array = np.array(mascara) / 255.0
            mascara_array = (mascara_array > 0.5).astype(np.float32)
            mascaras.append(mascara_array)
    
    imagenes = np.array(imagenes)
    mascaras = np.expand_dims(np.array(mascaras), axis=-1)
    return imagenes, mascaras

# Entrenamiento
ruta_imagenes_entrenamiento = r".\data\modelTrainingImages"
ruta_mascaras_entrenamiento = r".\data\modelMaskImages"

X_train_full, y_train_full = cargar_imagenes_y_mascaras(ruta_imagenes_entrenamiento, ruta_mascaras_entrenamiento)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

model = unet()
history = model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_val, y_val))

# Guardar el modelo
model.save(r".\data\model30_256.h5")
