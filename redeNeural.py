import numpy as np



#pesos Hidden(entre a de entrada e a escondida) e Out(entre a escondida e a de saida) 
#esse size eh o (tanto de neuronios,quantos pesos)
pesosH = np.random.uniform(0,1, size=(5,3))
pesosO = np.random.uniform(0,1, size=(4,1))

alfa = 7.0
#print pesosH
#print pesosO

bias = -1

dados = np.genfromtxt('iris.csv', delimiter=';')
#normalizing values
for i in range(5):
    dados[:,i] = ( dados[:,i] - min(dados[:,i]) ) / (max(dados[:,i]) - min(dados[:,i]))

Treino = list(dados[0:35])+ list(dados[50:85])+ list(dados[100:135])
Teste  = list(dados[35:50])+ list(dados[85:100])+ list(dados[135:150])





'''
PesosH:
[[0.05174497 0.68133367 0.87620812]
 [0.22281067 0.51856598 0.1992142 ]
 [0.71583844 0.25143159 0.47226234]
 [0.4846727  0.46988578 0.52115562]
 [0.49318576 0.67066392 0.10552139]] --> esse ultimo eh o vetor do bias

 entrada: 
 [0.12 0.34 075 0.87 -1]
 .
 .
 .

pesosO:
[[0.09410545]
 [0.33197879]
 [0.79257158]
 [0.49988089]]  --> esse ultimo eh o vetor do bias
 '''


def testaRede(teste):
	respostas = []
	entradaH = []
	entradaO = []
	##entre a camada de entrada e a camada escondida
	for x in teste:
		respostas.append(x[-1])
		entradaH.append(np.append(x[:-1], -1)) #adicionando o -1(bias) no final das entradas
	entradaH = np.array(entradaH)
	saidasH = sigmoide(np.dot(entradaH,pesosH))
	##entre a camada escondida e a camada de saida
	for x in saidasH:
		entradaO.append(np.append(x, -1)) #adicionando o -1(bias) no final das entradas
	entradaO = np.array(entradaO)
	saidasO = sigmoide(np.dot(entradaO,pesosO))
	return saidasO,saidasH,respostas,entradasH

def treinaRede(teste):
	while True:
		saidaO,saidaH,respostas,entradasH = testaRede(teste)
		respostas = np.reshape(respostas,(len(respostas),1))
		erro = respostas - saidaO
		if erro 
			break
		gradienteO = erro*sigmoideDerivada(saidaO)
		deltaO = alfa * gradienteO * saidasH
		gradienteH = gradienteO.dot(pesosO.T)*sigmoideDerivada(saidasH)
		deltaH = alfa * gradienteH.dot(entradasH.T)
		pesosH += deltaH
		pesosO += deltaO

def sigmoide(x):
	return 1.0 / (1 + np.exp(-x))

def sigmoideDerivada(x):
	return np.multiply(x,1-x)

def computaSinapse(entrada,neuronio,bias):
	for x in xrange(0,len(neuronio)):
		if x == len(neuronio)-1:
			total+=neuronio[x]*bias
		else:
			total += neuronio[x]*entrada[x]
	return sigmoide(total)

#testaRede(Teste)
treinaRede(Teste)