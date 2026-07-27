import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42

DATA_PATH = Path("/home/(user)/PG/python/data.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_PATH = MODEL_DIR / "mejor_modelo.pkl"
PARAMETERS_PATH = MODEL_DIR / "mejores_hiperparametros.json"
TREE_IMAGE_PATH = OUTPUT_DIR / "arbol_decision.png"
FITNESS_IMAGE_PATH = OUTPUT_DIR / "evolucion_fitness.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "matriz_confusion.png"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"

POPULATION_SIZE = 100
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80
NUM_GENERATIONS = 50
ELITE_SIZE = 2

FEATURES = [
    "DV",
    "VV",
    "PB",
    "Temp",
    "HR",
    "RS",
    "PM@10",
    "PM2@5",
    "OZONO",
    "SO2",
    "NO",
    "NO2",
    "NOX",
    "CO",
]


# ============================================================
# PREPARACIÓN DE DIRECTORIOS
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA Y PREPROCESAMIENTO DE DATOS
# ============================================================

def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """
    Carga el archivo CSV, convierte las variables requeridas a valores
    numéricos, elimina registros incompletos y genera las categorías.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {file_path.resolve()}"
        )

    data = pd.read_csv(file_path)

    missing_columns = [
        column for column in FEATURES if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "El archivo CSV no contiene las siguientes columnas requeridas: "
            + ", ".join(missing_columns)
        )

    # Convertir todas las características a formato numérico.
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Eliminar registros que tengan variables faltantes.
    rows_before = len(data)
    data = data.dropna(subset=FEATURES).copy()
    rows_after = len(data)

    print(f"Registros originales: {rows_before}")
    print(f"Registros válidos: {rows_after}")
    print(f"Registros eliminados: {rows_before - rows_after}")

    if data.empty:
        raise ValueError(
            "No quedaron registros válidos después del preprocesamiento."
        )

    data = create_environmental_categories(data)

    return data


def create_environmental_categories(data: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la variable objetivo Categoria_indice.

    Las condiciones se ordenan desde las categorías potencialmente más
    desfavorables hasta las más favorables para reducir conflictos cuando
    varias reglas se cumplen al mismo tiempo.
    """

    extremadamente_desfavorable = data["PM2@5"] >= 20

    desfavorable = (
        (data["SO2"] >= 0.1)
        | (data["NOX"] >= 0.1)
    )

    regular = (
        (data["Temp"] >= 30)
        | (data["HR"] >= 80)
    )

    muy_favorable = (
        (data["RS"] > 800)
        & (data["PM@10"] < 20)
    )

    buena = (
        (data["Temp"] < 25)
        & (data["HR"] >= 40)
        & (data["OZONO"] < 0.05)
    )

    razonable_buena = (
        (data["Temp"] < 30)
        & (data["HR"] < 70)
        & (data["OZONO"] < 0.1)
    )

    conditions = [
        extremadamente_desfavorable,
        desfavorable,
        regular,
        muy_favorable,
        buena,
        razonable_buena,
    ]

    categories = [
        "Extremadamente Desfavorable",
        "Desfavorable",
        "Regular",
        "Muy Favorable",
        "Buena",
        "Razonable Buena",
    ]

    data["Categoria_indice"] = np.select(
        conditions,
        categories,
        default="Sin Categoria",
    )

    return data


# ============================================================
# DIVISIÓN DE DATOS
# ============================================================

def split_data(data: pd.DataFrame):
    """
    Divide el conjunto de datos en:

    - 60% entrenamiento
    - 20% validación
    - 20% prueba

    El conjunto de validación se usa durante la evolución.
    El conjunto de prueba solamente se usa al final.
    """

    X = data[FEATURES]
    y = data["Categoria_indice"]

    stratify_target = y if y.value_counts().min() >= 2 else None

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    validation_stratify = (
        y_train_validation
        if y_train_validation.value_counts().min() >= 2
        else None
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=validation_stratify,
    )

    print("\nDistribución de los datos:")
    print(f"Entrenamiento: {len(X_train)}")
    print(f"Validación: {len(X_validation)}")
    print(f"Prueba: {len(X_test)}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# REPRESENTACIÓN DEL CROMOSOMA
# ============================================================

def create_random_chromosome() -> dict:
    """
    Genera una configuración aleatoria del árbol de decisión.
    """

    return {
        "max_depth": int(np.random.randint(1, 16)),
        "min_samples_split": int(np.random.randint(2, 21)),
        "min_samples_leaf": int(np.random.randint(1, 11)),
        "criterion": str(
            np.random.choice(["gini", "entropy", "log_loss"])
        ),
    }


# ============================================================
# FUNCIÓN DE EVALUACIÓN
# ============================================================

def evaluate_chromosome(
    chromosome: dict,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> float:
    """
    Entrena un árbol con los parámetros del cromosoma y devuelve
    su exactitud sobre el conjunto de validación.
    """

    model = DecisionTreeClassifier(
        max_depth=chromosome["max_depth"],
        min_samples_split=chromosome["min_samples_split"],
        min_samples_leaf=chromosome["min_samples_leaf"],
        criterion=chromosome["criterion"],
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return accuracy_score(y_validation, predictions)


# ============================================================
# OPERADORES GENÉTICOS
# ============================================================

def tournament_selection(
    population: list[dict],
    fitness_scores: list[float],
    tournament_size: int,
) -> dict:
    """
    Selecciona un cromosoma mediante torneo.
    """

    effective_size = min(tournament_size, len(population))

    tournament_indices = np.random.choice(
        len(population),
        size=effective_size,
        replace=False,
    )

    winner_index = max(
        tournament_indices,
        key=lambda index: fitness_scores[index],
    )

    return population[winner_index].copy()


def crossover(
    parent1: dict,
    parent2: dict,
    crossover_rate: float,
) -> tuple[dict, dict]:
    """
    Realiza cruzamiento uniforme entre dos cromosomas.
    """

    child1 = parent1.copy()
    child2 = parent2.copy()

    if np.random.random() > crossover_rate:
        return child1, child2

    for key in parent1:
        if np.random.random() < 0.5:
            child1[key], child2[key] = child2[key], child1[key]

    return child1, child2


def mutate(chromosome: dict, mutation_rate: float) -> dict:
    """
    Aplica mutación independiente a cada gen.

    A diferencia del código original, min_samples_split no cambia
    automáticamente en todas las mutaciones.
    """

    mutated = chromosome.copy()

    if np.random.random() < mutation_rate:
        mutated["max_depth"] = int(np.random.randint(1, 16))

    if np.random.random() < mutation_rate:
        mutated["min_samples_split"] = int(np.random.randint(2, 21))

    if np.random.random() < mutation_rate:
        mutated["min_samples_leaf"] = int(np.random.randint(1, 11))

    if np.random.random() < mutation_rate:
        mutated["criterion"] = str(
            np.random.choice(["gini", "entropy", "log_loss"])
        )

    return mutated


# ============================================================
# ALGORITMO GENÉTICO
# ============================================================

def run_genetic_algorithm(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> tuple[dict, float, list[float], list[float]]:
    """
    Ejecuta el algoritmo genético y devuelve:

    - Mejor cromosoma
    - Mejor exactitud
    - Historial de mejor exactitud
    - Historial de exactitud promedio
    """

    np.random.seed(RANDOM_STATE)

    population = [
        create_random_chromosome()
        for _ in range(POPULATION_SIZE)
    ]

    best_fitness_history = []
    average_fitness_history = []

    global_best_chromosome = None
    global_best_fitness = -np.inf

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [
            evaluate_chromosome(
                chromosome,
                X_train,
                X_validation,
                y_train,
                y_validation,
            )
            for chromosome in population
        ]

        sorted_indices = np.argsort(fitness_scores)[::-1]

        generation_best_index = sorted_indices[0]
        generation_best_fitness = fitness_scores[generation_best_index]
        generation_average_fitness = float(np.mean(fitness_scores))

        if generation_best_fitness > global_best_fitness:
            global_best_fitness = generation_best_fitness
            global_best_chromosome = population[
                generation_best_index
            ].copy()

        best_fitness_history.append(global_best_fitness)
        average_fitness_history.append(generation_average_fitness)

        print(
            f"Generación {generation + 1:02d}/{NUM_GENERATIONS} | "
            f"Mejor validación: {generation_best_fitness:.4f} | "
            f"Mejor global: {global_best_fitness:.4f} | "
            f"Promedio: {generation_average_fitness:.4f}"
        )

        # Elitismo: conservar los mejores cromosomas.
        new_population = [
            population[index].copy()
            for index in sorted_indices[:ELITE_SIZE]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(
                population,
                fitness_scores,
                TOURNAMENT_SIZE,
            )

            parent2 = tournament_selection(
                population,
                fitness_scores,
                TOURNAMENT_SIZE,
            )

            child1, child2 = crossover(
                parent1,
                parent2,
                CROSSOVER_RATE,
            )

            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    # Evaluar correctamente la población final.
    final_fitness_scores = [
        evaluate_chromosome(
            chromosome,
            X_train,
            X_validation,
            y_train,
            y_validation,
        )
        for chromosome in population
    ]

    final_best_index = int(np.argmax(final_fitness_scores))
    final_best_fitness = final_fitness_scores[final_best_index]
    final_best_chromosome = population[final_best_index].copy()

    if final_best_fitness > global_best_fitness:
        global_best_fitness = final_best_fitness
        global_best_chromosome = final_best_chromosome

    return (
        global_best_chromosome,
        global_best_fitness,
        best_fitness_history,
        average_fitness_history,
    )


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN FINAL
# ============================================================

def train_final_model(
    best_chromosome: dict,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
) -> DecisionTreeClassifier:
    """
    Entrena el modelo final utilizando los conjuntos de entrenamiento
    y validación combinados.
    """

    X_final_train = pd.concat(
        [X_train, X_validation],
        axis=0,
    )

    y_final_train = pd.concat(
        [y_train, y_validation],
        axis=0,
    )

    model = DecisionTreeClassifier(
        max_depth=best_chromosome["max_depth"],
        min_samples_split=best_chromosome["min_samples_split"],
        min_samples_leaf=best_chromosome["min_samples_leaf"],
        criterion=best_chromosome["criterion"],
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_final_train, y_final_train)

    return model


def evaluate_final_model(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Evalúa el modelo final sobre datos de prueba no utilizados durante
    la evolución.
    """

    predictions = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\nEvaluación final")
    print("=" * 60)
    print(f"Exactitud en prueba: {test_accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(f"Exactitud en prueba: {test_accuracy:.4f}\n\n")
        file.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45,
    )

    display.figure_.set_size_inches(12, 8)
    display.figure_.tight_layout()
    display.figure_.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(display.figure_)

    return test_accuracy


# ============================================================
# GRÁFICAS Y GUARDADO
# ============================================================

def save_model(
    model: DecisionTreeClassifier,
    best_chromosome: dict,
    validation_accuracy: float,
    test_accuracy: float,
) -> None:
    """
    Guarda el modelo y sus metadatos.
    """

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "best_chromosome": best_chromosome,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "genetic_algorithm": {
            "population_size": POPULATION_SIZE,
            "tournament_size": TOURNAMENT_SIZE,
            "mutation_rate": MUTATION_RATE,
            "crossover_rate": CROSSOVER_RATE,
            "num_generations": NUM_GENERATIONS,
            "elite_size": ELITE_SIZE,
        },
    }

    with open(PARAMETERS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nModelo guardado en: {MODEL_PATH.resolve()}")
    print(f"Parámetros guardados en: {PARAMETERS_PATH.resolve()}")


def plot_fitness_history(
    best_history: list[float],
    average_history: list[float],
) -> None:
    """
    Grafica la evolución de la función de aptitud.
    """

    generations = range(1, len(best_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        generations,
        best_history,
        label="Mejor exactitud",
    )
    plt.plot(
        generations,
        average_history,
        label="Exactitud promedio",
    )

    plt.xlabel("Generación")
    plt.ylabel("Exactitud")
    plt.title("Evolución del algoritmo genético")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FITNESS_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Gráfica de evolución guardada en: "
        f"{FITNESS_IMAGE_PATH.resolve()}"
    )


def plot_decision_tree_model(
    model: DecisionTreeClassifier,
) -> None:
    """
    Guarda la visualización del árbol de decisión.
    """

    plt.figure(figsize=(28, 16))

    plot_tree(
        model,
        filled=True,
        rounded=True,
        feature_names=FEATURES,
        class_names=model.classes_.tolist(),
        fontsize=7,
        proportion=True,
    )

    plt.title("Árbol de decisión optimizado mediante algoritmo genético")
    plt.tight_layout()

    plt.savefig(
        TREE_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Árbol guardado en: {TREE_IMAGE_PATH.resolve()}")


# ============================================================
# PREDICCIÓN DE NUEVOS DATOS
# ============================================================

def predict_new_observation(
    model: DecisionTreeClassifier,
    values: dict,
) -> str:
    """
    Predice la categoría de una nueva observación.
    """

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in values
    ]

    if missing_features:
        raise ValueError(
            "Faltan variables para realizar la predicción: "
            + ", ".join(missing_features)
        )

    new_observation = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    prediction = model.predict(new_observation)

    return str(prediction[0])


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    data = load_and_prepare_data(DATA_PATH)

    print("\nDistribución de categorías:")
    print(data["Categoria_indice"].value_counts())

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(data)

    (
        best_chromosome,
        best_validation_accuracy,
        best_fitness_history,
        average_fitness_history,
    ) = run_genetic_algorithm(
        X_train,
        X_validation,
        y_train,
        y_validation,
    )

    print("\nMejor cromosoma encontrado:")
    print(best_chromosome)

    print(
        "Mejor exactitud de validación: "
        f"{best_validation_accuracy:.4f}"
    )

    best_model = train_final_model(
        best_chromosome,
        X_train,
        X_validation,
        y_train,
        y_validation,
    )

    test_accuracy = evaluate_final_model(
        best_model,
        X_test,
        y_test,
    )

    save_model(
        best_model,
        best_chromosome,
        best_validation_accuracy,
        test_accuracy,
    )

    plot_fitness_history(
        best_fitness_history,
        average_fitness_history,
    )

    plot_decision_tree_model(best_model)

    # Ejemplo de predicción.
    #
    # new_day = {
    #     "DV": 180,
    #     "VV": 3.5,
    #     "PB": 1012,
    #     "Temp": 27,
    #     "HR": 45,
    #     "RS": 700,
    #     "PM@10": 18,
    #     "PM2@5": 9,
    #     "OZONO": 0.04,
    #     "SO2": 0.02,
    #     "NO": 0.01,
    #     "NO2": 0.03,
    #     "NOX": 0.04,
    #     "CO": 0.5,
    # }
    #
    # prediction = predict_new_observation(
    #     best_model,
    #     new_day,
    # )
    #
    # print(
    #     "\nPronóstico de calidad ambiental:",
    #     prediction,
    # )


if __name__ == "__main__":
    main()
