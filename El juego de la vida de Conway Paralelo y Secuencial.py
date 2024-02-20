import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count, Lock
import time

# Declaración global del mutex
mutex = Lock()

def vecinos(i, j, rows, cols):
    return [(i+1, j), (i+1, j+1), (i, j+1), (i-1, j+1),
            (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1)]

def contar_vecinos(viva, rows, cols):
    contador = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            for v in vecinos(i, j, rows, cols):
                if 0 <= v[0] < rows and 0 <= v[1] < cols:
                    contador[i, j] += viva[v[0], v[1]]
    return contador

def siguiente_generacion(submatriz, rows, cols):
    nueva_generacion = np.zeros(submatriz.shape, dtype=int)
    contador = contar_vecinos(submatriz, submatriz.shape[0], cols)
    for i in range(submatriz.shape[0]):
        for j in range(cols):
            if submatriz[i, j] == 1:
                if contador[i, j] < 2 or contador[i, j] > 3:
                    nueva_generacion[i, j] = 0
                else:
                    nueva_generacion[i, j] = 1
            elif contador[i, j] == 3:
                nueva_generacion[i, j] = 1
    return nueva_generacion

def generar_estado_inicial(rows, cols):
    return np.random.choice([0, 1], size=(rows, cols))

def juego_vida_secuencial(rows, cols, num_generaciones):
    estado_actual = generar_estado_inicial(rows, cols)
    tiempo_inicio = time.time()
    generaciones = []
    for _ in range(num_generaciones):
        estado_actual = siguiente_generacion(estado_actual, rows, cols)
        generaciones.append(np.copy(estado_actual))
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    return tiempo_ejecucion, generaciones

# Función auxiliar para envolver el cálculo de siguiente_generacion con un mutex
def siguiente_generacion_wrapper(submatriz, rows, cols):
    with mutex:
        return siguiente_generacion(submatriz, rows, cols)

def juego_vida_paralelo(rows, cols, num_generaciones):
    estado_actual = generar_estado_inicial(rows, cols)
    tiempo_inicio = time.time()
    generaciones = []
    for _ in range(num_generaciones):
        submatrices = []
        for i in range(cpu_count()):
            subfilas = rows // cpu_count()
            submatriz = estado_actual[i * subfilas: (i + 1) * subfilas, :]
            submatrices.append((submatriz, subfilas, cols))
        
        with Pool(cpu_count()) as pool:
            resultado = pool.starmap(siguiente_generacion_wrapper, submatrices)
        
        estado_actual = np.concatenate(resultado)
        generaciones.append(np.copy(estado_actual))
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    return tiempo_ejecucion, generaciones

def update(frame, ax, generaciones_secuencial, generaciones_paralelo):
    ax[0].clear()
    ax[0].matshow(generaciones_secuencial[frame], cmap='binary')
    ax[0].set_title(f'Secuencial: {frame+1}')

    ax[1].clear()
    ax[1].matshow(generaciones_paralelo[frame], cmap='binary')
    ax[1].set_title(f'Paralelo - Generación: {frame+1}')

if __name__ == "__main__":
    filas = 1000
    columnas = 1000
    num_generaciones = 10

    print("Ejecutando la versión secuencial...")
    tiempo_secuencial, generaciones_secuencial = juego_vida_secuencial(filas, columnas, num_generaciones)
    print(f"Tiempo de ejecución secuencial para {num_generaciones} generaciones: {tiempo_secuencial} segundos")

    rendimiento_secuencial = num_generaciones / tiempo_secuencial
    print(f"Rendimiento Secuencial: {rendimiento_secuencial}")

    print("Ejecutando la versión paralela...")
    tiempo_paralelo, generaciones_paralelo = juego_vida_paralelo(filas, columnas, num_generaciones)
    print(f"Tiempo de ejecución paralela para {num_generaciones} generaciones: {tiempo_paralelo} segundos")

    rendimiento_paralelo = num_generaciones / tiempo_paralelo
    print(f"Rendimiento Paralelo: {rendimiento_paralelo}")

    num_cpus = cpu_count()
    print("Número de CPUs usados:", num_cpus)

    speed_up = tiempo_secuencial / tiempo_paralelo
    print(f"Speed-up: {speed_up}")

    fig, ax = plt.subplots(1, 2)

    anim = FuncAnimation(fig, update, frames=num_generaciones, fargs=(ax, generaciones_secuencial, generaciones_paralelo), repeat=False)

    plt.show()