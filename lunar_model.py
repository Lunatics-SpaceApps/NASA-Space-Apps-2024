# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from script import training_data, test_data
from skimage.transform import resize
import tensorflow as tf

# %%
d1 = r'.\data\lunar\training\data\S12_GradeA'
d2 = r'.\data\lunar\training\catalogs'
d3 = r'.\data\lunar\test\data\S15_GradeA'

xtrn, ytrn = training_data(d1, d2)
xtst = test_data(d3)

# %%
# Suponiendo que xtrn y xtst son tus matrices de velocidades
# xtrn tiene forma (n_sismos_train, n_tiempos)
# xtst tiene forma (n_sismos_test, n_tiempos)

n_sismos_train, n_tiempos_train = xtrn.shape
n_sismos_test, n_tiempos_test = xtst.shape
spectrogram_train_list = []
spectrogram_test_list = []

# Parámetros del filtro
minfreq = 0.5
maxfreq = 1.0

# Frecuencia de muestreo (ajusta según tus datos)
fs = 6.625

# Calcular el espectrograma para cada sismo en el conjunto de entrenamiento
for i in range(n_sismos_train):
    # Obtener las velocidades de un sismo (una fila de xtrn)
    velocities = xtrn[i, :]
    
    # Calcular el espectrograma para estas velocidades
    f, t, sxx = signal.spectrogram(velocities, fs=fs)
    
    # Guardar el espectrograma (sxx) en una lista
    spectrogram_train_list.append(sxx)

# Calcular el espectrograma para cada sismo en el conjunto de prueba
for i in range(n_sismos_test):
    # Obtener las velocidades de un sismo (una fila de xtst)
    velocities = xtst[i, :]
    
    # Calcular el espectrograma para estas velocidades
    f, t, sxx = signal.spectrogram(velocities, fs=fs)
    
    # Guardar el espectrograma (sxx) en una lista
    spectrogram_test_list.append(sxx)

# Convertir las listas a matrices numpy si es necesario
spectrogram_train = np.array(spectrogram_train_list)  # Forma (n_sismos_train, frequencies, times)
spectrogram_test = np.array(spectrogram_test_list)    # Forma (n_sismos_test, frequencies, times)

# Verificar las formas de las matrices de espectrogramas
print("Shape of Training Spectrograms:", spectrogram_train.shape)
print("Shape of Testing Spectrograms:", spectrogram_test.shape)


# %%
# Suponiendo que spectrogram_train y spectrogram_test tienen las formas correctas
n_sismos_train, freqs_train, times_train = spectrogram_train.shape
n_sismos_test, freqs_test, times_test = spectrogram_test.shape

# Definir el nuevo tamaño al que quieres reducir los espectrogramas (por ejemplo, 64x64)
new_freqs = 128
new_times = 500

# Crear una lista para almacenar los espectrogramas redimensionados del conjunto de entrenamiento
spectrograms_resized_train = []

# Redimensionar cada espectrograma del conjunto de entrenamiento
for i in range(n_sismos_train):
    resized_sxx = resize(spectrogram_train[i], (new_freqs, new_times), mode='reflect', anti_aliasing=True)
    spectrograms_resized_train.append(resized_sxx)

# Convertir la lista en una matriz numpy
spectrograms_resized_train = np.array(spectrograms_resized_train)

# Crear una lista para almacenar los espectrogramas redimensionados del conjunto de prueba
spectrograms_resized_test = []

# Redimensionar cada espectrograma del conjunto de prueba
for i in range(n_sismos_test):
    resized_sxx = resize(spectrogram_test[i], (new_freqs, new_times), mode='reflect', anti_aliasing=True)
    spectrograms_resized_test.append(resized_sxx)

# Convertir la lista en una matriz numpy
spectrograms_resized_test = np.array(spectrograms_resized_test)

# Verificar las formas de los espectrogramas redimensionados
print("Shape of Resized Training Spectrograms:", spectrograms_resized_train.shape)
print("Shape of Resized Testing Spectrograms:", spectrograms_resized_test.shape)

# Visualizar un espectrograma redimensionado del conjunto de entrenamiento
plt.figure(figsize=(10, 5))
plt.pcolormesh(np.linspace(0, new_times, new_times), np.linspace(0, new_freqs, new_freqs), spectrograms_resized_train[20], shading='gouraud')
plt.title('Resized Training Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power')
plt.show()

# Visualizar un espectrograma redimensionado del conjunto de prueba
plt.figure(figsize=(10, 5))
plt.pcolormesh(np.linspace(0, new_times, new_times), np.linspace(0, new_freqs, new_freqs), spectrograms_resized_test[5], shading='gouraud')
plt.title('Resized Testing Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power')
plt.show()


# %%
# Definir el modelo CNN
model = tf.keras.models.Sequential([
    # Primera capa convolucional
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(new_freqs, new_times, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Segunda capa convolucional
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Tercera capa convolucional
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Aplanar las características para pasarlas a una capa densa
    tf.keras.layers.Flatten(),

    # Capa densa completamente conectada
    tf.keras.layers.Dense(128, activation='relu'),

    # Capa de salida
    tf.keras.layers.Dense(1, activation='linear')  # Salida con una sola neurona para la predicción del valor final (regresión)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Resumen del modelo
model.summary()


# %%
# Normalizar los espectrogramas a un rango [0, 1]
spectrograms_resized_train = (spectrograms_resized_train - np.min(spectrograms_resized_train)) / (np.max(spectrograms_resized_train) - np.min(spectrograms_resized_train))
# Normalizar las etiquetas a un rango [0, 1]
max_y = np.max(ytrn)  # Debes tener los valores originales para esto
min_y = np.min(ytrn)
ytrn = (ytrn - np.min(ytrn)) / (np.max(ytrn) - np.min(ytrn))

# Asegurarse de que los espectrogramas tengan la forma (n_sismos, 64, 64, 1)
X = spectrograms_resized_train[..., np.newaxis]

# Entrenar el modelo
model.fit(X, ytrn, epochs=10, batch_size=16, validation_split=0.2)

# %%
n_samples_test = spectrograms_resized_test.shape[0]
spectrograms_resized_test = (spectrograms_resized_test - np.min(spectrograms_resized_test)) / (np.max(spectrograms_resized_test) - np.min(spectrograms_resized_test))
spectrograms_input = spectrograms_resized_test.reshape(n_samples_test, new_freqs, new_times, 1)

# 2. Realizar las predicciones
predictions = model.predict(spectrograms_input)
predictions_original = predictions * (max_y - min_y) + min_y

# 3. Ver las predicciones
print("Predictions:", predictions_original)

# %%
i = 0  # Índice del espectrograma que quieres visualizar
time_to_mark = predictions_original[0]

# Graficar velocidades en función del tiempo
plt.figure(figsize=(10, 3))
plt.plot(range(len(xtst[i, :])), xtst[i, :], label='Velocity')
plt.axvline(x=time_to_mark*len(xtst[i, :]), color='red', linestyle='--', label=f'Detection t={time_to_mark * len(xtst[i, :])} sec')
plt.xlabel('Time [sec]')
plt.ylabel('Velocity [m/s]')
plt.title('Seismic')
plt.grid(True)
plt.legend()
plt.show()

# Visualizar un espectrograma redimensionado del conjunto de prueba con su etiqueta
plt.figure(figsize=(10, 5))
plt.pcolormesh(np.linspace(0, new_times, new_times), np.linspace(0, new_freqs, new_freqs), spectrograms_resized_test[i], shading='gouraud')
# Añadir la línea vertical roja en el tiempo especificado
plt.axvline(x=time_to_mark*500, color='red', linestyle='--')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power')
plt.show()


