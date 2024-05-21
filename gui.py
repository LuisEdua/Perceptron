import tkinter as tk
from tkinter import filedialog
import pandas as pd
import perceptron  # Asegúrate de que este módulo esté correctamente referenciado

file_path = ""

def cargar_csv():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])

def iniciar_entrenamiento():
    global file_path
    if file_path:
        df = pd.read_csv(file_path, delimiter=';', header=None)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        p_or = perceptron.Perceptron(int(entry_epocas.get()), float(entry_error.get()), float(entry_tasa.get()), df)
        p_or.fit()
        perceptron.plot_evolution(p_or)

# Crear la ventana principal
ventana = tk.Tk()
ventana.geometry("400x350")
ventana.title("Cargar CSV")
ventana.config(bg = "#2D3250")

# Campos de entrada para los parámetros del perceptrón
entry_error = tk.Entry(ventana)
entry_error.pack(pady=5)
entry_tasa = tk.Entry(ventana)
entry_tasa.pack(pady=5)
entry_epocas = tk.Entry(ventana)
entry_epocas.pack(pady=5)

# Etiquetas para los campos de entrada
tk.Label(ventana, text="Error permisible", bg = "#424669", fg = "white", font =("Arial", 11, "bold"), ).pack(before=entry_error, pady = 5)
tk.Label(ventana, text="Tasa de aprendizaje" , bg = "#424669", fg = "white", font =("Arial", 11, "bold") ).pack(before=entry_tasa, pady = 5)
tk.Label(ventana, text="Número de épocas", bg = "#424669", fg = "white", font =("Arial", 11, "bold")).pack(before=entry_epocas, pady = 5)


# Crear un botón para cargar el archivo CSV
boton_cargar = tk.Button(ventana, text="Cargar CSV", command=cargar_csv, highlightthickness=0, borderwidth=0, fg="black", bg = "#676F9D" ,font=("Arial", 12))
boton_cargar.pack(pady=20)

# Botón para iniciar el entrenamiento
boton_entrenar = tk.Button(ventana, text="Iniciar entrenamiento", command=iniciar_entrenamiento, highlightthickness=0, borderwidth=0, fg="black",  bg = "#F2B077" , font=("Arial", 12))
boton_entrenar.pack(pady=20)

# Iniciar el bucle principal de la aplicación
ventana.mainloop()
