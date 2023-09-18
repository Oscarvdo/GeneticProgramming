import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
 
import matplotlib.pyplot as plt 

# Carga los datos
data = pd.read_csv('/home/(user)/PG/python/data.csv')

# Corrige los valores no numéricos en las columnas requeridas
columns_to_convert = ['Temp', 'HR', 'OZONO', 'RS', 'PM@10', 'PM2@5', 'SO2', 'NOX']
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Definir las condiciones para las categorías de índice
# Ajusta estos valores según tus criterios
buena_condicion = (data['Temp'] < 25) & (data['HR'] >= 40) & (data['OZONO'] < 0.05)
razonable_buena_condicion = (data['Temp'] < 30) & (data['HR'] < 70) & (data['OZONO'] < 0.1)
regular_condicion = (data['Temp'] >= 30) | (data['HR'] >= 80)
desfavorable_condicion = (data['SO2'] >= 0.1) | (data['NOX'] >= 0.1)
muy_favorable_condicion = (data['RS'] > 800) & (data['PM@10'] < 20)
extremadamente_desfavorable_condicion = (data['PM2@5'] >= 20)

# Asignar las categorías en función de las condiciones
data['Categoria_indice'] = np.select([
    buena_condicion,
    razonable_buena_condicion,
    regular_condicion,
    desfavorable_condicion,
    muy_favorable_condicion,
    extremadamente_desfavorable_condicion
], [
    'Buena',
    'Razonable Buena',
    'Regular',
    'Desfavorable',
    'Muy Favorable',
    'Extremadamente Desfavorable'
], default='Sin Categoria')

# Seleccionar características y variable objetivo
features = ['DV', 'VV', 'PB', 'Temp', 'HR', 'RS', 'PM@10', 'PM2@5', 'OZONO', 'SO2', 'NO', 'NO2', 'NOX', 'CO']
X = data[features]
y = data['Categoria_indice']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la representación de la solución (cromosoma)
# En este caso, no se utilizan coeficientes, sino hiperparámetros del árbol de decisión.



def create_random_chromosome():
    # Genera un cromosoma con hiperparámetros aleatorios
    max_depth = np.random.randint(1, 10)  # Profundidad máxima del árbol
    min_samples_split = np.random.randint(2, 10)  # Mínimo de muestras para dividir (modificado)
    min_samples_leaf = np.random.randint(1, 10)  # Mínimo de muestras en una hoja
    chromosome = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    return chromosome

# Crear una función de evaluación
def evaluate_chromosome(chromosome, X_train, X_test, y_train, y_test):
    # Crea un modelo de árbol de decisión con los hiperparámetros del cromosoma
    model = DecisionTreeClassifier(max_depth=chromosome['max_depth'], min_samples_split=chromosome['min_samples_split'],
                                   min_samples_leaf=chromosome['min_samples_leaf'], random_state=42)
    
    # Entrena el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)
    
    # Realiza predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcula la métrica de evaluación (exactitud)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Implementar operadores genéticos (selección, cruzamiento y mutación)

def selection(population, fitness_scores, tournament_size):
    selected_indices = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]  # Selecciona el ganador basado en la exactitud máxima
        selected_indices.append(winner_index)
    return selected_indices

def crossover(parent1, parent2):
    # Implementa el cruzamiento aquí (intercambiar valores entre los padres para crear dos nuevos cromosomas)
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Intercambia valores de hiperparámetros entre los padres
    for key in parent1:
        if np.random.rand() < 0.5:
            child1[key], child2[key] = child2[key], child1[key]
    
    return child1, child2

def mutate(chromosome, mutation_rate):
    # Implementa la mutación aquí (cambia aleatoriamente algunos valores de hiperparámetros)
    mutated_chromosome = chromosome.copy()
    for key in mutated_chromosome:
        if key == 'min_samples_split':
            mutated_chromosome[key] = np.random.randint(2, 10)  # Asegura que min_samples_split sea válido
        elif np.random.rand() < mutation_rate:
            mutated_chromosome[key] = np.random.randint(1, 10)  # Cambiar aleatoriamente el valor
    return mutated_chromosome

# Ejemplo de uso de las funciones:
population_size = 100
tournament_size = 5
mutation_rate = 0.1

# Genera una población inicial de cromosomas
population = [create_random_chromosome() for _ in range(population_size)]

# Evolución de la población
num_generations = 50

for generation in range(num_generations):
    fitness_scores = [evaluate_chromosome(chromosome, X_train, X_test, y_train, y_test) for chromosome in population]
    
    # Realiza la selección
    selected_indices = selection(population, fitness_scores, tournament_size)
    new_population = []
    
    for i in range(0, population_size, 2):
        parent1 = population[selected_indices[i]]
        parent2 = population[selected_indices[i + 1]]
        
        # Aplica cruzamiento
        child1, child2 = crossover(parent1, parent2)
        
        # Aplica mutación
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        new_population.extend([child1, child2])
    
    population = new_population

# En este punto, population contiene la última generación de cromosomas
# Puedes seleccionar el mejor cromosoma (hiperparámetros óptimos) y entrenar tu modelo con él.

best_chromosome = population[np.argmax(fitness_scores)]  # Selecciona el mejor cromosoma basado en la exactitud máxima

# Entrena un modelo de árbol de decisión con los hiperparámetros óptimos en el conjunto completo de entrenamiento
best_model = DecisionTreeClassifier(max_depth=best_chromosome['max_depth'], min_samples_split=best_chromosome['min_samples_split'],
                                   min_samples_leaf=best_chromosome['min_samples_leaf'], random_state=42)
best_model.fit(X_train, y_train)

# Guarda el mejor modelo en un archivo
joblib.dump(best_model, 'mejor_modelo.pkl')

model_path = 'mejor_modelo.pkl'
print(os.path.abspath(model_path))
# Cargar el modelo entrenado
best_model = joblib.load('mejor_modelo.pkl')

# Definir los datos del día en el que deseas hacer la predicción
# Asegúrate de que los datos tengan las mismas características que se utilizaron para entrenar el modelo
#nuevo_dia = np.array([DV_value, VV_value, PB_value, Temp_value, HR_value, RS_value, PM10_value, PM25_value, OZONO_value, SO2_value, NO_value, NO2_value, NOX_value, CO_value]).reshape(1, -1)

# Realiza la predicción utilizando el modelo
#pronostico = best_model.predict(nuevo_dia)

# 'pronostico' contendrá la categoría pronosticada para la calidad del aire en el nuevo día
#print("Pronóstico de Calidad del Aire para el Nuevo Día:", pronostico[0])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Reemplaza 'best_model' con el nombre de tu modelo entrenado
model = best_model

plt.figure(figsize=(12, 6))
plot_tree(best_model, filled=True, feature_names=features, class_names=best_model.classes_.tolist())  # Convierte las clases a una lista
plt.savefig('arbol_decision.png')  # Guarda el árbol como imagen
