from random import random
import numpy as np
import matplotlib.pyplot as plt
from math import tanh
from math import pi
from matplotlib.animation import FuncAnimation

#definiremos os objetos neurônio, camada e rede:

class Neuronio:
  def __init__(self, npesos, taxa_aprendizado = 0.002):
    
    self.taxa_aprendizado = taxa_aprendizado
    self.bias = random() #o bias é inicializado como um valor aleatório entre 0 e 1
    self.pesos = []
    for i in range(npesos):
        self.pesos.append(2*random() - 1) #os pesos são inicializados como valores aleatórios entre -1 e 1

  def ativacao(self):
    return tanh(self.saida_desativada)

  def derivada_ativacao(self):
    return 1 - self.ativacao()**2

  def saida(self, entradas):
    saida = self.bias 
    for i in range(len(self.pesos)):
        saida += self.pesos[i]*entradas[i]
    self.saida_desativada = saida #guardamos a saida desativada, que será usada na ativação e na aprendizagem
    self.entradas = entradas #guardamos as entradas
    return self.ativacao()
  
  def aprendeneuronio(self, dloss_dsaida, novo_erro):
    dloss_dsaidadesativada = dloss_dsaida*(self.derivada_ativacao())

    for i in range(len(self.pesos)):
      novo_erro[i] += dloss_dsaidadesativada*self.pesos[i]
      #a dloss/dsaida do neuronio i da camada anterior é a soma das dloss/dentrada dos neuronios da camada atual
      self.pesos[i] -= dloss_dsaidadesativada*self.entradas[i]*self.taxa_aprendizado
      #subtraimos (dloss/dpeso)*taxa_aprendizado de cada peso

    self.bias -= dloss_dsaidadesativada*self.taxa_aprendizado #subtraimos (dloss/dbias)*taxa_aprendizado do bias

class Camada:
  def __init__(self, tamanho, tamanhocamadaanterior):
    self.camada = [] #a camada em si é um vetor de neuronios
    for i in range(tamanho):
        self.camada.append(Neuronio(tamanhocamadaanterior))

  def resultado_camada(self, entrada):
    resultado = [] #o resultado da camada é um vetor com o resultado de cada neuronio
    for neuronio in self.camada:
        resultado.append(neuronio.saida(entrada))
    return resultado

  def aprendecamada(self, dloss_dsaida_camada): #treina os neuronios da camada e devolve um vetor  para treinar as proxs
    dloss_dsaida_camada_anterior = [0]*len(self.camada[0].pesos)
    for neuronio, dloss_dsaida in zip(self.camada, dloss_dsaida_camada):
      Neuronio.aprendeneuronio(neuronio, dloss_dsaida, dloss_dsaida_camada_anterior)
    return dloss_dsaida_camada_anterior 

  
class Rede:
  def __init__(self, nentradas, nsaidas, ncamadasocultas, largura):
    self.numero_de_neuronios = nentradas + nsaidas + largura*ncamadasocultas 
    self.numero_de_camadas = 2 + ncamadasocultas
    self.rede = [] #a rede em si é um vetor de camadas
    self.rede.append(Camada(largura, nentradas))
    for i in range(1, ncamadasocultas):
        self.rede.append(Camada(largura,largura))
    self.rede.append(Camada(nsaidas, largura))
  
  def resultado_rede(self, entrada):
    atual = entrada
    for camada in self.rede:
      atual = camada.resultado_camada(atual) #aplicamos resultado camada sobre o resultado da camada anterior
    return atual #atual será, nesse ponto, a saída final da rede

  def aprende_rede(self, entrada, esperado):
    obtido = self.resultado_rede(entrada)
    mean_squared_error = 0
    derivadas_individuais = []
    n = len(obtido)
    for o,e in zip(obtido, esperado):
      derivadas_individuais.append(2/n * (o-e)) #2(o-e)/n é dloss/dsaidafinal
      mean_squared_error += ((o-e)**2)/n
    self.mean_squared_error = mean_squared_error #a rede terá um parâmetro associado ao MSE relativo a entrada e ao esperado
    dloss_dsaida_camada = derivadas_individuais
    for i in range(len(self.rede)-1, -1, -1): #aplicaremos aprendecamada começando do fim e indo em direção a entrada
      dloss_dsaida_camada = Camada.aprendecamada(self.rede[i], dloss_dsaida_camada)

rede = Rede(1, 1, 2, 3) #Criamos uma rede de nome rede
epoca = 0 #o contador de épocas começa no 0

fig = plt.figure(figsize=(8,7))

plt.title(f"Aprendizado de uma cossenoide em uma rede com {rede.numero_de_neuronios} neurônios em {rede.numero_de_camadas} camadas")
x = np.arange(0, 2*pi, 0.6)
y = np.cos(x)
plt.scatter(x, y, label="Pontos de treino")
plt.xlabel(f"Época: {epoca}")

entrada = np.arange(0, 2*pi, 0.01)
plt.plot(entrada, np.cos(entrada), color='black', label="Cossenoide")

line, = plt.plot([], [], label="Curva prevista pela rede neural")

def animate(i):
    for i in range(20):
      for n in range(len(x)):
        rede.aprende_rede([x[n]], [y[n]])
    global epoca
    epoca = epoca + 20
    plt.xlabel(f"Época: {epoca}     Erro quadrático médio: %.3f     Função de ativação: tanh     Taxa de aprendizado: {rede.rede[0].camada[0].taxa_aprendizado}" % rede.mean_squared_error)

    previsto = []
    for i in entrada:
        previsto.append(rede.resultado_rede([i]))
    line.set_data(entrada, previsto)
    return line

animacao = FuncAnimation(fig, animate, frames=200, interval=10)
plt.legend(loc="upper center")
animacao.save('redeneuralanimada.gif', dpi=120)