from scipy.fft import fft

# Funcion para simular el esquema de modulacion

def modulador16(bits, fc, mpp): 

    # 1. Se definen los parametros de la señal
    N = len(bits) # Cantidad de bits

    # 2. Se construye un periodo de la señal
    Ts = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Ts, mpp)  # mpp: muestras por período
    portadora1 = np.sin(2*np.pi*fc*t_periodo)
    portadora2 = np.sin(2*np.pi*fc*t_periodo)

    # 3. Se inicializa la señal modulada
    t_simulacion = np.linspace(0, N*Ts, N*mpp) 
    senal_Tx1 = np.zeros(t_simulacion.shape)
    senal_Tx2 = np.zeros(t_simulacion.shape)
    moduladora1 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    moduladora2 = np.zeros(t_simulacion.shape)
 
    # 4. Se asigna la forma de la onda segun los diferentes bits
    for i, bit in enumerate(bits):
        if i%4 == 0:
            if bits[i] == 0 and bits[i+1] == 0:
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1 * -3
                moduladora1[i*mpp : (i+1)*mpp] = -3
            if bits[i] == 0 and bits[i+1] == 1:
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1 * -1
                moduladora1[i*mpp : (i+1)*mpp] = -1
            if bits[i] == 1 and bits[i+1] == 1:
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1 * 1
                moduladora1[i*mpp : (i+1)*mpp] = 1
            if bits[i] == 1 and bits[i+1] == 0:
                senal_Tx1[i*mpp : (i+1)*mpp] = portadora1 * 3
                moduladora1[i*mpp : (i+1)*mpp] = 3
                
            if bits[i+2] == 0 and bits[i+3] == 0:
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2 * 3
                moduladora2[i*mpp : (i+1)*mpp] = -3
            if bits[i+2] == 0 and bits[i+3] == 1:
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2 * 1
                moduladora2[i*mpp : (i+1)*mpp] = -1
            if bits[i+2] == 1 and bits[i+3] == 1:
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2 * -1
                moduladora2[i*mpp : (i+1)*mpp] = -1
            if bits[i+2] == 1 and bits[i+3] == 0:
                senal_Tx2[i*mpp : (i+1)*mpp] = portadora2 * -3
                moduladora2[i*mpp : (i+1)*mpp] = -3
        else:
            continue
    
    # 5. Se calcula la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Ts)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx1, senal_Tx2, P_senal_Tx, portadora1, moduladora1, moduladora2  

import numpy as np


# Funcion que simula un medio de transmision no ideal con ruido AWGN

def canal_ruidoso16(senal_Tx1, senal_Tx2, Pm, SNR):

    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido1 = np.random.normal(0, np.sqrt(Pn), senal_Tx1.shape)
    ruido2 = np.random.normal(0, np.sqrt(Pn), senal_Tx2.shape)
    # Señal distorsionada por el canal ruidoso
    senal_Rx1 = senal_Tx1 + ruido1
    senal_Rx2 = senal_Tx2 + ruido2
    return senal_Rx1, senal_Rx2

# Funcion que simula un bloque demodulador de señales con esquema BPSK

def demodulador16(senal_Rx1, senal_Rx2, portadora, mpp):

    # Cantidad de muestras en senal_Rx
    muestra1 = len(senal_Rx1)
    muestra2 = len(senal_Rx2)
    # Cantidad de bits (símbolos) en transmisión
    cant_bits = int(muestra1 / mpp)
    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(cant_bits)

    # Vector para la señal demodulada
    senal_demodulada1 = np.zeros(senal_Rx1.shape)
    senal_demodulada2 = np.zeros(senal_Rx2.shape)
    
    # Pseudo-energía de un período de la portadora
    Es = np.sum(portadora * portadora)

    # Demodulación
    for i in range(cant_bits):
        # Producto interno de dos funciones
        producto1 = senal_Rx1[i*mpp : (i+1)*mpp] * portadora
        Ep1 = np.sum(producto1) 
        senal_demodulada1[i*mpp : (i+1)*mpp] = producto1
        producto2 = senal_Rx2[i*mpp : (i+1)*mpp] * portadora
        Ep2 = np.sum(producto2) 
        senal_demodulada2[i*mpp : (i+1)*mpp] = producto2


        # Criterio de decisión por detección de energía
        if i % 4 == 0:
            if Ep1 >= 2*Es:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
            if Ep1 <= 2*Es and Ep1 > 0:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
            if Ep1 >= -2*Es and Ep1 < 0:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
            if Ep1 <= -2*Es:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
            
            if Ep2 >= 2*Es:
                bits_Rx[i+2] = 0
                bits_Rx[i+3] = 0
            if Ep2 <= 2*Es and Ep2 > 0:
                bits_Rx[i+2] = 0
                bits_Rx[i+3] = 1
            if Ep2 >= -2*Es and Ep2 < 0:
                bits_Rx[i+2] = 1
                bits_Rx[i+3] = 1
            if Ep2 <= -2*Es:
                bits_Rx[i+2] = 1
                bits_Rx[i+3] = 0
                
        else: 
            continue

    return bits_Rx.astype(int), senal_demodulada1, senal_demodulada2

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Ts = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Ts / mpp

# Tiempo de la simulación
T = Ns * Ts

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros:
# Frecuencia de mas de 3k de la portadora
# Muestras de la portadora
# Relacion señal-ruido

fc = 4000  
mpp = 20  
SNR = -5 

# Iniciar medición del tiempo de simulación
inicio1 = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx1 = fuente_info('arenal.jpg')
dimensiones1 = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx1 = rgb_a_bit(imagen_Tx1)

# 3. Modular la cadena de bits usando el esquema 16-QAM
senal_Tx1, senal_Tx2, Pm, portadora, moduladora1, moduladora2 = modulador16(bits_Tx1, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx1, senal_Rx2 = canal_ruidoso16(senal_Tx1, senal_Tx2, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx16, senal_demodulada1, senal_demodulada2 = demodulador16(senal_Rx1, senal_Rx2, portadora, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx16 = bits_a_rgb(bits_Rx16, dimensiones1)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular errores
errores = sum(abs(bits_Tx1 - bits_Rx16))
BER = errores/len(bits_Tx1)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx1)
ax.set_title('Transmitido')

# Mostrar imagen
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx16)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx16)




#PARTE 3. CALCULO DE LA DENSIDAD ESPECTRAL DE POTENCIA
from scipy.fft import fft

    
# Transformada de Fourier
senal_f = fft(senal_Tx1 + senal_Tx2)

# Muestras de la señal
Nm = len(senal_Tx1 + senal_Tx2)

#Número de simbolos
Ns = Nm // mpp

#Tiempo del símbolo = periodo de la onda portadora
Ts = 1 / fc

#Tiempo entre muestras (período de muestreo)
Tm = Ts / mpp

#Tiempo de la simulación
T = Ns * Ts

#Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm),Nm//2)

#Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]),2))
plt.xlim(0, 20000)
plt.grid()
plt.show()
