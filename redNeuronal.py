
#Entrenamiento de una red neuronal para realizar la compuerta xor

import numpy as np
import matplotlib.pyplot as plt

class perceptron:

    def __init__(self, n):
        self.pesos = np.random.randn(n)  #arroja un vector con 3 valores aleatorios 
        self.n = n                       #asigna a la variable global n el valor de la variable local n que en este caso es 3
        self.vecsalida=[0,0,0,0,0]
    def propagacion(self, entradas):
        
        self.salida = 1 * (self.pesos.dot(entradas) > 0)  #para cada valor de entrada verifica si el valor es mayor que 0; si es mayor lo multiplica por 1
        self.entradas = entradas                          #asigna a la variable global entradas el valor de la variable local entradas

    def actualizacion(self, alfa, salida):

        for i in range(0, self.n):  # ciclo que va desde 0 hasta n=3
            """
                Pesos es un vector de tamaño 3
                alfa=.5

            """
            self.pesos[i] = self.pesos[i] + alfa * (salida - self.salida)* self.entradas[i]  #Asigna un nuevo peso para cada una de los vectores de entrada

perceptron_and = perceptron(3)  # instancia la clase perceptron y asigna el valor 3 a las variable n en __init__
ejemplos = np.array([[0,0,0,0],[0,1,0,1],[0,1,1,0],[1,0,1,0],[1,1,0,0]]) # se crear los valores de entrada y salida

grad_pesos = [perceptron_and.pesos]  # recibe el vector de valores aleatorios guardados en pesos
for epoca in range(0, 100): 
    for i in range(0, 4):
        perceptron_and.propagacion(ejemplos[i,0:3])
        perceptron_and.actualizacion(0.5, ejemplos[i,3])
        grad_pesos = np.concatenate((grad_pesos, [perceptron_and.pesos]), axis = 0)
        perceptron_and.vecsalida[i]=perceptron_and.salida
    print('epoca: ', epoca ,' salida:')
    print(perceptron_and.vecsalida[:])

plt.plot(grad_pesos[:,0],'k')
plt.plot(grad_pesos[:,1],'r')
plt.plot(grad_pesos[:,2],'b')
plt.show()
