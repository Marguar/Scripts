import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def FitInterval(X, Y, start, finish):
    x = X[start:finish+1]
    y = Y[start:finish+1]
    plt.plot(X, Y)
    coeff = np.polyfit(x, y, 1)

    # Plot dei dati utilizzati per il fit
    plt.plot(x, y)

    # Disegno della retta sul grafico
    y_fit = coeff[0]*x + coeff[1]
    plt.plot(x, y_fit)

    return coeff[0], coeff[1]

def getdata(filename, dataLines=[6, float("inf")]):
    # Set up the Import Options and import the data
    opts = pd.read_csv(filename, delimiter="\t", skiprows=dataLines[0]-1)
    
    # Select the desired columns
    Time = opts["Time"]
    Amplitude = opts["Amplitude"]
    
    return Time, Amplitude

def getNdata(filename, dataLines=[2, 2]):
    # Set up the Import Options and import the data
    opts = pd.read_csv(filename, delimiter="\t", skiprows=dataLines[0]-1, nrows=dataLines[1]-dataLines[0]+1)
    # Select the desired column
    VarName4 = opts["VarName4"]
    return VarName4

def plotch(x, y, xu, yu, n):
    plt.plot(x, y)
    plt.xlabel(xu)
    plt.ylabel(yu)
    plt.title('Channel ' + str(n))
    plt.show()

def plotfunc(x, y, xu, yu, n):
    plt.plot(x, y)
    plt.xlabel(xu)
    plt.ylabel(yu)
    plt.title('Func ' + str(n))
    plt.show()

#Funzione utile per l'import dei dati, così da poter escludere una parte di essi, soprattutto agli edge
def seleziona_dati(dati, inizio, fine, colonna):
    return dati[inizio: fine, colonna]

# Definisci la funzione della corrente critica (Ic) con la formula di Ambegaokar-Baratoff
def IcRn_fit(T, A, Delta_0, Tc):
    k_B = 8.617333262145e-05 # Costate Boltzmann [eV/K]
    e = 1.602176634e-19  # Carica elementare [C]
    Delta = Delta_0 * np.tanh(1.74 * np.sqrt(Tc / T - 1))
    return (A*np.pi * Delta) / (2 * e) * np.tanh(Delta / (2 * k_B * T))