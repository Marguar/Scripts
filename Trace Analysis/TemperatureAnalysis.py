# Il seguente script fa un'analisi delle curve corrente tensione in temperatura
# Formato file IV_meas_T=0.010K, altrimenti cambiare la parte nel ciclo for
import os
import matplotlib.pyplot as plt
import numpy as np
import Fun
from scipy.optimize import curve_fit

# Definisci le cartelle
cartella_dati = r"C:\Users\nefra\OneDrive\Tesi Magistrale\Programmazione\Python\Data\Alluminio_30nm\SIS_20C\IVT_longrange"
cartella_analisi=r"C:\Users\nefra\OneDrive\Tesi Magistrale\Programmazione\Python\Data\Alluminio_30nm\SIS_20C\Analisi"

# Inizializza variabili e dizionari
Vsoglia = 2e-5
Fitpoints= 1000
SinglePlotName="SIS_LongRange"
startData=1000
endData=None

# Preallocazione
temperature = []
Ic=[]
Rn=[]
misura = {
    'V': [],
    'I': []
}

# Scansiona tutti i file nella cartella dei dati
for nome_file in os.listdir(cartella_dati):

    # Estrai il valore della Temp dal nome del file
    Temp = float(nome_file.split("T=")[-1].split("K")[0])
    temperature.append(Temp)

    # Carica i dati da file
    dati = np.loadtxt(os.path.join(cartella_dati, nome_file))
    V = Fun.seleziona_dati(dati, startData, endData, 0)
    I = Fun.seleziona_dati(dati, startData, endData, 1)

    # Calcola I critica in base alla soglia messa
    cond=V>Vsoglia
    indx=np.argmax(cond)
    Ic.append(I[indx])

    #Salva Dati
    misura['V'].append(V)
    misura['I'].append(I)
    
    # Plot della V nel Tempo
    plt.plot(V)
    plt.scatter(indx,V[indx])
    plt.xlabel('Time (arb)')
    plt.ylabel('Voltage (V)')
    plt.title(f"V vs t {SinglePlotName} T = {Temp} K")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(cartella_analisi,f"{SinglePlotName}_V_{Temp}_K.png" ))
    plt.close()

    # Plot della I nel Tempo
    plt.plot(I)
    plt.scatter(indx,I[indx])
    plt.xlabel('Time (arb)')
    plt.ylabel('Current (A)')
    plt.title(f"I vs t {SinglePlotName} T = {Temp} K")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(cartella_analisi,f"{SinglePlotName}_I_{Temp}_K.png" ))
    plt.close()

    # Plot singoli delle curve I vs V
    plt.plot(V, I)
    plt.scatter(V[indx],I[indx])
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title(f"I vs V {SinglePlotName} T = {Temp} K")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(cartella_analisi,f"{SinglePlotName}_{Temp}_K.png" ))
    plt.close()

    # Linear fit in base al numero di point specificati
    start = np.where(I == np.max(I))[0][0]
    finish = start + Fitpoints
    G, a = Fun.FitInterval(V, I, start, finish)
    R = 1 / G
    Rn.append(R) 
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title(f"I vs V {SinglePlotName} T = {Temp} K\nResistenza: {format(R, '.2e')} Ohm")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig(os.path.join(cartella_analisi,f"{SinglePlotName}_Fit_{Temp}_K.png" ))
    plt.close()

# Grafici All in One IV 
for Temp,voltage,current in zip(temperature,misura['V'],misura['I']):
    plt.plot(voltage,current,label="T = " + str(Temp))
    plt.legend(fontsize='small',prop={'size': 8},ncol=2)

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title(f"I vs V {SinglePlotName}")
plt.xlabel('Voltage(V)')
plt.ylabel('Current(A)')
plt.savefig(os.path.join(cartella_analisi,f"I vs V {SinglePlotName}.png" ))

# Crea delle linee che fanno vedere la soglia
plt.axvline(x=Vsoglia, color='red', linestyle='--')
plt.axvline(x=-Vsoglia, color='red', linestyle='--')
plt.savefig(os.path.join(cartella_analisi,f"I vs V Soglia {SinglePlotName}.png" ))
plt.close()

# Grafico Ic vs T
plt.plot(temperature, Ic)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title(f"Ic vs T {SinglePlotName}")
plt.xlabel('Temperature (K)')
plt.ylabel('Current(A)')
plt.savefig(os.path.join(cartella_analisi,f"Ic vs T {SinglePlotName}.png" ))
plt.close()

# Grafico Rn vs T
plt.plot(temperature, Rn)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title(f"Ic vs T {SinglePlotName}")
plt.xlabel('Temperature (K)')
plt.ylabel('Normal Resistance (Ω)')
plt.savefig(os.path.join(cartella_analisi,f"Rn vs T {SinglePlotName}.png" ))
plt.close()

# Calcolo degli Ic*Rn
Ic=np.array(Ic)
Rn=np.array(Rn)
IcRn=Ic*Rn
T_data = np.array(temperature)

# Grafico IcRn vs T
plt.plot(temperature, IcRn,'o',label='Experimental Data')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel('Temperature (K)')
plt.ylabel('Voltage (V)')
plt.title(f"IcRn vs T {SinglePlotName}")
plt.savefig(os.path.join(cartella_analisi,f"IcRn vs T {SinglePlotName}.png" ))

# Definisci i limiti dei parametri A, Delta_0, Tc
bounds = ([-np.inf, -np.inf, 1.301], [np.inf, np.inf, 1.4])

# Effettua il fit dei dati sperimentali alla funzione IcRn_fit
initial_guess = [1e-20, 190e-6, 1.35]  # Valori iniziali stimati per i parametri A, Delta_0, Tc
fit_params, fit_covariance = curve_fit(Fun.IcRn_fit, T_data, IcRn, p0=initial_guess, bounds=bounds)

# Estrai i parametri del fit
A_fit, Delta_0_fit, Tc_fit = fit_params
A_error, Delta_0_error, Tc_error = np.sqrt(np.diag(fit_covariance))

# Stampa i parametri del fit con gli errori
print("Parametri del fit:")
print("A =", A_fit, "±", A_error)
print("Delta_0 =", Delta_0_fit, "±", Delta_0_error)
print("Tc =", Tc_fit, "±", Tc_error)

# Plot dei dati Fit
x=np.linspace(0.001,1.3,10000)
y=Fun.IcRn_fit(x, A_fit, Delta_0_fit, Tc_fit)
plt.plot(x,y,label='Fit Data')
plt.title(f"IcRn vs T {SinglePlotName} \nA: {format(A_fit, '.2e')}  Δ₀: {format(Delta_0_fit, '.2e')}  Tc: {format(Tc_fit, '.2e')}")
plt.legend(fontsize='small',prop={'size': 8})
plt.savefig(os.path.join(cartella_analisi,f"IcRn vs T FIT{SinglePlotName}.png" ))
plt.show()
plt.close()