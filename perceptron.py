import numpy as np
import matplotlib.pyplot as plt

step = lambda x: np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, epochs, error_permisible, tasa_aprendizaje, df):
        self.epochs = epochs
        self.error_permisible = error_permisible
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values.reshape(-1, 1)
        self.n_inputs = df.shape[1] - 1
        self.bias = 1
        self.weights = np.random.rand(self.n_inputs + 1, 1)
        self.error_history = []
        self.weight_history = [self.weights.flatten().tolist()]
        self.act_f = step
        self.tasa_aprendizaje = tasa_aprendizaje

    def predict(self, X):
        np.dot(X, self.weights)
        return self.act_f(np.dot(X, self.weights))

    def fit(self):
        n_correct_predictions = 0
        for _ in range(self.epochs):
            errors = []
            X_with_bias = np.hstack([np.ones((self.X.shape[0], self.bias)), self.X])
            for x, target in zip(X_with_bias, self.y):
                output = self.predict(x.reshape(1, -1))
                error = target - output
                self.weights += self.tasa_aprendizaje * error * x.reshape(-1, 1)
                errors.append(abs(error.flatten()[0]))
            self.weight_history.append(self.weights.flatten().tolist())
            self.error_history.append(np.mean(errors))
            if np.mean(errors) > self.error_permisible:
                n_correct_predictions = 0
            else:
                n_correct_predictions += 1
            if n_correct_predictions == 5:
                self.epochs = len(self.error_history)
                break


def plot_evolution(perceptron):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    primeros = ""
    for i in range(len(perceptron.weight_history[0])):
        primeros += f"w{i}:{perceptron.weight_history[0][i]} | "
    primeros += "\n"

    ultimos = ""
    for i in range(len(perceptron.weight_history[-1])):
        ultimos += f"w{i}:{perceptron.weight_history[-1][i]} | "
    ultimos += "\n"

    axs[0].axis('off')
    valores = f"""Primeros pesos: {primeros}
    Últimos pesos: {ultimos}
    Tasa de aprendizaje: {perceptron.tasa_aprendizaje}
    Error permisible: {perceptron.error_permisible}
    Número de iteraciones (épocas): {perceptron.epochs}"""
    axs[0].text(0.5, 0.5, valores, fontsize=12, ha='center')

    weights = np.array(perceptron.weight_history)
    for i in range(weights.shape[1]):
        axs[1].plot(weights[:, i], label=f'w{i}')
    axs[1].set_title('Evolución de los pesos')
    axs[1].set_xlabel('Época')
    axs[1].set_ylabel('Valor del peso')
    axs[1].legend()

    epochs = list(range(1, perceptron.epochs + 1))
    axs[2].plot(epochs, perceptron.error_history, label='Error')
    axs[2].set_title('Evolución del error durante el entrenamiento')
    axs[2].set_xlabel('Época')
    axs[2].set_ylabel('Error')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
