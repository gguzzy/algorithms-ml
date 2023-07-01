import numpy as np

"""
1. Utilizzare la classe usando 
# supponiamo di avere 4 esempi di allenamento (n=4), ciascuno con 3 caratteristiche (N=3)
X = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9],
              [1.0, 1.1, 1.2]])

# ciascun esempio di allenamento ha un output corrispondente
y = np.array([0.2, 0.6, 1.0, 1.4])

# Aggiungiamo una colonna di 1 per il nostro bias a X
# la creiamo della stessa altezza di X ed aggiunga 1 colonna di 1
bias = np.ones((X.shape[0], 1))
# axis = 1 lungo asse delle colonne
X = np.concatenate((bias, X), axis=1)
"""


class WidrowHoff:
    """
    Implementazione della regola di apprendimento di Widrow-Hoff (o Regola Delta)

    Durante la chiamata al costruttore inizializzare i seguenti campi, ove necessario:
    :param X: Matrice di input con le caratteristiche degli esempi.
    :param y: Vettore di output desiderati.
    :param learning_rate: Velocità di apprendimento o passo dell'algoritmo. Opzionale, di default è 0.01.
    :param tol: Tolleranza per il criterio di arresto. Opzionale, di default è 1e-4.
    :return: Un'istanza del modello Widrow-Hoff.

    """

    # Definisco il costruttore della mia classe con parametri da inizializzare
    def __init__(self, X, y, learning_rate=0.01, tol=1e-4):
        """
        :param X: Matrice di input su cui effettuare le previsioni.
        :return: Le previsioni del modello sugli input.
        """
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.tol = tol
        # initializzo i pesi
        # ritorno numero righe .shape[0]
        # ritorno numero colonne .shape[1]
        self.weights = np.random.rand(self.X.shape[1])

    # predicts the output
    def predict(self, X):
        """
        Calcola le previsioni del modello sugli input forniti.
        :param X: Matrice di input su cui effettuare le previsioni.
        :return: Le previsioni del modello sugli input.
        """
        return np.dot(X, self.weights)

    def update_weights(self):
        """
        Aggiorna i pesi del modello in base all'errore commesso sul set di addestramento.
        """
        # calcolo gli output, chiamo funzione predict
        outputs = self.predict(self.X)
        # Update the weights: aggiorno effettivamente qui i pesi
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot((self.y - outputs), self.X[:, i])

    def train(self):
        """
        Esegue l'addestramento del modello fino alla convergenza, definita come una variazione dell'errore quadratico
        totale minore della tolleranza fornita.
        """
        SSEold = np.inf
        while True:
            # aggiorno i dati
            self.update_weights()

            # calcolo errore e calclo nuovo SSEnew
            outputs = self.predict(self.X)
            SSEnew = np.sum((self.y - outputs) ** 2)

            # Controllo la convergenza
            if abs(SSEnew - SSEold < self.tol):
                break
            else:
                SSEold = SSEnew


# Esempio di utilizzo della classe WidrowHoff con un set di dati sintetici
np.random.seed(0)  # Per la riproducibilità dei risultati.
N = 100  # Numero di esempi di addestramento.
d = 2  # Numero di caratteristiche per ciascun esempio.

# Generiamo caratteristiche casuali e pesi veri.
X = np.random.normal(size=(N, d))
true_weights = np.random.normal(size=(d + 1))  # +1 per il termine di bias.

# Calcoliamo gli output corrispondenti.
y = np.dot(np.concatenate((np.ones((N, 1)), X), axis=1), true_weights)

# Creiamo una istanza della classe WidrowHoff e alleniamo il modello.
model = WidrowHoff(X, y, tol=1e-4)
model.train()

# Calcoliamo l'errore quadrato medio finale.
predictions = model.predict(X)
mse = np.mean((y - predictions) ** 2)
print(f"Errore Quadrato Medio finale: {mse}")

# Stampiamo i pesi veri e i pesi appresi dal modello.
print(f"Pesi veri: {true_weights}")
print(f"Pesi appresi: {model.weights}")
print("Final model weights: ", model.weights)
