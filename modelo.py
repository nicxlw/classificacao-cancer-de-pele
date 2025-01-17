import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


#------LEITURA E TRATAMENTO DE DADOS---------

# Ler o dataframe de metadados
df = pd.read_csv('HAM10000_metadata.csv')

# Colocar os caminhos das imagens no dataframe
df['path'] = df['image_id'].map(lambda x: os.path.join('HAM10000_images/', x + '.jpg'))

# Função para carregar e processar as imagens
def carregar_imagens(dataframe, tamanho=(32, 32)):
    imagens = [] # Vetor com as imagens convertidas
    rotulos = [] # Vetor com os rótulos das respectivas imagens
    for _, linha in dataframe.iterrows(): # Para cada linha do dataframe
        try:
            # Carregar imagem, redimensionar e converter para array
            img = Image.open(linha['path']).resize(tamanho)
            img_array = np.asarray(img) / 255.0  # Transformar em matriz de pixels e normalizar para [0, 1]
            imagens.append(img_array)
            rotulos.append(linha['dx'])
        except Exception as e:
            print(f"Erro ao carregar a imagem: {linha['path']}, erro: {e}")
    return np.array(imagens), np.array(rotulos)

# Mapear rótulos categóricos para valores numéricos
rotulos_num = {rotulo: num for num, rotulo in enumerate(df['dx'].unique())}
num_rotulos = {num: rotulo for rotulo, num in rotulos_num.items()}

# Balanceando a quantidade de fotos
datagen = ImageDataGenerator( # Definindo especificações do gerador de imagens
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Limite de imagens por categoria
quantidade = 2000  # Número fixo de imagens por categoria
imagens_balanceadas = []
rotulos_balanceados = []

# Iterar sobre cada categoria
for categoria in tqdm(df['dx'].unique(), desc="Processando categorias"):
    # Filtrar imagens da categoria atual
    categoria_img = df[df['dx'] == categoria]
    imagens, rotulos = carregar_imagens(categoria_img)  # Usa a função anterior
    
    # Se exceder a quantidade, sorteia [quantidade] amostras
    if len(imagens) > quantidade:
        indices = np.random.choice(len(imagens), quantidade, replace=False)
        imagens = imagens[indices]
    
    # Adicionar imagens e seus rótulos às listas
    imagens_balanceadas.extend(imagens)
    rotulos_balanceados.extend([rotulos_num[categoria]] * len(imagens))
    
    # Gerar imagens adicionais se necessário
    if len(imagens) < quantidade:
        # Número de imagens que faltam
        deficit = quantidade - len(imagens)
        
        # Criar gerador para essa categoria
        gen = datagen.flow(
            np.array(imagens),
            np.array([rotulos_num[categoria]] * len(imagens)),  # Rótulos da categoria
            batch_size=1
        )
        
        # Gerar imagens e rótulos adicionais e adicioná-los às listas
        for _ in range(deficit):
            img, _ = gen.next()
            imagens_balanceadas.append(img[0])
            rotulos_balanceados.append(rotulos_num[categoria])

# Converter listas em arrays NumPy
imagens_balanceadas = np.array(imagens_balanceadas)
rotulos_balanceados = np.array(rotulos_balanceados)

# Plotar nova distribuição de dados
contagem = pd.Series(rotulos_balanceados).map(num_rotulos).value_counts()
plt.figure(figsize=(10, 5))
contagem.plot(kind='bar', color='lightblue')
plt.title("Distribuição de imagens após balanceamento")
plt.xlabel("Categoria")
plt.ylabel("Quantidade de imagens")
plt.show()


#---------MODELO---------
classes = len(rotulos_num)
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    imagens_balanceadas, rotulos_balanceados, test_size=0.2, random_state=42, stratify=rotulos_balanceados
)

# Converter os rótulos para one-hot encoding (evita hierarquias)
y_train_onehot = to_categorical(y_train, num_classes=classes)
y_test_onehot = to_categorical(y_test, num_classes=classes)

# Criar o modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

# Compilar o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar modelo
history = modelo.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=20,
    batch_size=32
)

#------------RESULTADOS-----------
perda, acuracia = modelo.evaluate(X_test, y_test_onehot)
print(f"Acurácia no conjunto de teste: {acuracia:.2f}")

# Gráficos
# Acurácia
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Perda
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Fazer predições no conjunto de teste
y_pred_probs = modelo.predict(X_test)  # Probabilidades preditas
y_pred = np.argmax(y_pred_probs, axis=1)  # Classe predita

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(rotulos_num.keys()))
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title("Matriz de Confusão")
plt.show()

# Definição das categorias
classes = ['bkl', 'nv', 'df', 'mel', 'vas', 'bcc', 'akiec']
categorias = {
    'sem_risco': ['bkl', 'nv', 'vas', 'df'],
    'risco': ['mel', 'bcc', 'akiec']
}

# Obter índices das categorias
sem_risco_indices = [classes.index(cat) for cat in categorias['sem_risco']]
risco_indices = [classes.index(cat) for cat in categorias['risco']]

# Cálculo dos erros
total_sem_risco = conf_matrix[sem_risco_indices, :].sum()  # Total de categorias sem risco
total_risco = conf_matrix[risco_indices, :].sum()  # Total de categorias de risco

# Casos de risco classificados como sem risco
fn = conf_matrix[np.ix_(risco_indices, sem_risco_indices)].sum()
# Casos sem risco classificados como de risco
fp = conf_matrix[np.ix_(sem_risco_indices, risco_indices)].sum()

# Exibir resultados
print(f"Porcentagem de fotos de categorias de risco classificadas como sem risco: {(fn / total_risco) * 100:.2f}%")
print(f"Porcentagem de fotos de categorias sem risco classificadas como risco: {(fp / total_sem_risco) * 100:.2f}%")

#-----EXEMPLO------
amostras_aleatorias = random.sample(range(len(X_test)), 3)

# Mapeamento de índices para categorias e riscos
categoria_real = [num_rotulos[y_test[i]] for i in amostras_aleatorias]
categoria_prevista = [num_rotulos[y_pred[i]] for i in amostras_aleatorias]
risco_real = ['risco' if cat in categorias['risco'] else 'sem_risco' for cat in categoria_real]
risco_previsto = ['risco' if cat in categorias['risco'] else 'sem_risco' for cat in categoria_prevista]

# Exibir as informações
print(" -- Exemplos de resultados -- ")
for i, indice in enumerate(amostras_aleatorias):
    print(f"Imagem {indice + 1}:")
    print(f"  Categoria real: {categoria_real[i]} ({risco_real[i]})")
    print(f"  Categoria prevista: {categoria_prevista[i]} ({risco_previsto[i]})")
    print()
