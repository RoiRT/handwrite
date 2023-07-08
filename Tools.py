import numpy as np
import matplotlib.pyplot as plt
from variables import caracteres

def cargar_datos(dataset):
    x_train = []
    y_train = []
    for data in dataset:
        x_train.append(np.fliplr(np.rot90(data['image'].numpy(), k=3)))
        y_train.append(data['label'].numpy())
    return np.array(x_train)/255.0, np.array(y_train).reshape(-1, 1)

def graficar_imagenes_generadas(i, generator):
    noise = np.random.normal(0, 1, [1, 100])

    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    images = generator.predict([np.tile(noise, (47, 1)), np.arange(47).reshape(-1, 1)])

    for j, image in enumerate(images):
        ax = axes[j // 10, j % 10]
        ax.imshow(image*255, cmap='gray')
        ax.set_title(caracteres[j])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("Evolution/grafica"+str(i)+".png", dpi=600)
    plt.close()