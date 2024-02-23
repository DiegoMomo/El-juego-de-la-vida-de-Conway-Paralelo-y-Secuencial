import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count
import time

def vecinos(i, j):
    return [(i+1, j), (i+1, j+1), (i, j+1), (i-1, j+1),
            (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1)]

def contar_vecinos(matriz, rows, cols):
    contador = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            for v in vecinos(i, j):
                if 0 <= v[0] < rows and 0 <= v[1] < cols:
                    contador[i, j] += matriz[v[0], v[1]]
    return contador

def siguiente_generacion(submatriz, rows, cols):
    nueva_generacion = np.zeros(submatriz.shape, dtype=int)
    contador = contar_vecinos(submatriz, submatriz.shape[0], cols)
    for i in range(submatriz.shape[0]):
        for j in range(submatriz.shape[1]):
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

def juego_vida_paralelo(rows, cols, num_generaciones):
    estado_actual = generar_estado_inicial(rows, cols)
    procesos = cpu_count()
    tiempo_inicio = time.time()
    generaciones = []
    for _ in range(num_generaciones):
        submatrices = []
        for i in range(procesos):
            subfilas = rows // procesos
            submatriz = estado_actual[i * subfilas: (i + 1) * subfilas, :]
            submatrices.append((submatriz, subfilas, cols))
        
        with Pool(procesos) as pool:
            resultado = pool.starmap(siguiente_generacion, submatrices)
        
        estado_actual = np.concatenate(resultado)
        generaciones.append(np.copy(estado_actual))
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    return tiempo_ejecucion, generaciones

def update_secuencial(frame, ax, generaciones_secuencial):
    ax.clear()
    ax.matshow(generaciones_secuencial[frame], cmap='binary')
    ax.set_title(f'Secuencial: {frame+1}')

def update_paralelo(frame, ax, generaciones_paralelo):
    ax.clear()
    ax.matshow(generaciones_paralelo[frame], cmap='binary')
    ax.set_title(f'Paralelo: {frame+1}')

if __name__ == "__main__":
    filas = 2000
    columnas = 2000
    num_generaciones = 3

    print()
    print("Ejecutando la versión secuencial...")
    tiempo_secuencial, generaciones_secuencial = juego_vida_secuencial(filas, columnas, num_generaciones)
    print(f"Tiempo de ejecución secuencial para {num_generaciones} generaciones: {tiempo_secuencial} segundos")

    rendimiento_secuencial = num_generaciones / tiempo_secuencial
    print(f"Throughput (Secuencial): {rendimiento_secuencial} generaciones por segundo")
    print()

    fig, ax = plt.subplots(1, 1)  
    anim = FuncAnimation(fig, update_secuencial, frames=num_generaciones, fargs=(ax, generaciones_secuencial), repeat=False)

    plt.show()

    print("Ejecutando la versión paralela...")
    tiempo_paralelo, generaciones_paralelo = juego_vida_paralelo(filas, columnas, num_generaciones)
    print(f"Tiempo de ejecución paralela para {num_generaciones} generaciones: {tiempo_paralelo} segundos")

    rendimiento_paralelo = num_generaciones / tiempo_paralelo
    print(f"Throughput (Paralelo): {rendimiento_paralelo} generaciones por segundo")

    num_cpus = 6
    print("Número de procesadores", num_cpus)

    speed_up = tiempo_secuencial / tiempo_paralelo
    print(f"Speed-up: {speed_up}")

    efficiency = (speed_up / num_cpus) * 100 
    print(f"Efficiency: {efficiency} %")  

    fig, ax = plt.subplots(1, 1) 
    anim = FuncAnimation(fig, update_paralelo, frames=num_generaciones, fargs=(ax, generaciones_paralelo), repeat=False)

    plt.show()
