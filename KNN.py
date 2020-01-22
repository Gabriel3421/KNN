
'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''
import math
import re
from scipy import stats
import numpy as np

vet_atributos = []#vetor de entradas
vet_respostas = []#vetor de saidas
one_out = 0 #qual amostra do vetor eu tou removendo
valor_tirado_att = np.ones((1, 4))#amostra retirada para teste
valor_tirado_resp = np.ones((1, 3))#resposta da amostra retirada para teste
vizinhos = 10 #quantidade de vizinhos
cont = 0
x = 0

def normaliza(x):
    #estou separando cada atributo em um vetor diferente e dps os normalizo e crio o
    #vetor com todos os os atributoss normalizados na msm ordem
    vet = list()
    vet1 = list()
    vet2 = list()
    vet3 = list()
    vet4 = list()

    cont1 = 0
    cont2 = 1
    cont3 = 2
    cont4 = 3

    for i in range(0,len(x)):
        if cont1 == i:
            vet1.append(x[i])
            cont1 = cont1 + 4
        if cont2 == i:
            vet2.append(x[i])
            cont2 = cont2 + 4
        if cont3 == i:
            vet3.append(x[i])
            cont3 = cont3 + 4
        if cont4 == i:
            vet4.append(x[i])
            cont4 = cont4 + 4
  
    vet1 = norm(vet1)    
    vet2 = norm(vet2)    
    vet3 = norm(vet3)    
    vet4 = norm(vet4)

    for i  in range(0,len(vet1)):
        vet.append(vet1[i])
        vet.append(vet2[i])
        vet.append(vet3[i])
        vet.append(vet4[i])

    return vet    

def norm(x):
    '''
    #testes usando outros tipo de normalizaçao a que deu resultados melhores foi 
    #a x[i]/max(x)
    vet = list()
    for i in range(0, len(x)):
        #aux = 2*((x[i] - min(x))/max(x)-min(x)) - 1
        aux = x[i]/max(x)
        vet.append(aux)
    return vet'''
    return stats.zscore(x)

dados = open("iris_log.dat", "r")

for line in dados:
    #separando o que é x do que é d 
    line = line.strip()#quebra no \n
    line = re.sub('\s+',',',line)#trocando os espaços vazios por virgula   
    a1,a2,a3,a4,r1,r2,r3 = line.split(",")#quebra nas virgulas e retorna 2 valores
    vet_atributos.append(float(a1))
    vet_atributos.append(float(a2))
    vet_atributos.append(float(a3))
    vet_atributos.append(float(a4))
    vet_respostas.append(float(r1))
    vet_respostas.append(float(r2))
    vet_respostas.append(float(r3))
dados.close()
#se for nomalizado acerta 142 amostras se nao acerta 145
vet_atributos = normaliza(vet_atributos)

def cria_mat_atributos(vet_atributos):
    #crio a matriz de atributos retirando umas das amostras da base de dados 
    #para ser o teste o valor do teste é salvo na variavel valor_tirado_att 
    # e é retornado o uma matriz com todas as outras amostras
    
    global one_out
    k=0
    vet = np.ones((int(len(vet_atributos)/4), 4))

    vet_1 = np.ones((int(len(vet_atributos)/4)-1, 4))

    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0,4):
            vet[i][j] = vet_atributos[k]
            k+=1

    for q in range(0,4):
           valor_tirado_att[0][q] =  vet[one_out][q]

    aux = list()
    for i in range(0, int(len(vet_atributos)/4)):
        for j in range(0,4):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/4)):
        for j in range(0,4):
            vet_1[i][j] = aux[k]
            k+=1
    
    return vet_1   

def cria_mat_resposta(vet_resposta):
    #crio a matriz de respostas retirando uma das amostras da base de dados 
    #para ser a resposta do valor do teste é salvo na variavel valor_tirado_resp 
    # e é retornado o uma matriz com todas as outras respostas das amostras
    
    global one_out
    k=0
    vet = np.ones((int(len(vet_resposta)/3), 3))

    vet_1 = np.ones((int(len(vet_resposta)/3)-1, 3))

    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0,3):
            vet[i][j]= vet_resposta[k]
            k+=1

    for q in range(0,3):
           valor_tirado_resp[0][q] =  vet[one_out][q]
    

    aux = list()
    for i in range(0, int(len(vet_resposta)/3)):
        for j in range(0,3):
            if i != one_out:
                aux.append(vet[i][j])
    k = 0
    for i in range(0, int(len(aux)/3)):
        for j in range(0,3):
            vet_1[i][j] = aux[k]
            k+=1
    #incrementando para na prox iteraçao retirar outra amostra da minha base
    one_out += 1        
    return vet_1

def calcula_dist(vet_atributo, valor_tirado_att):
    #calcula a distancia da amostra de teste para todas as outras 
    #amostras e retorna a matriz de atributos com uma coluna a mais 
    #que é a distancia dakela amostra a amostra teste 

    mat = np.zeros((len(vet_atributo), 5))
    for i in range(0, len(vet_atributo)):
        for j in range(0,4):
            mat[i][j] = vet_atributo[i][j]
    
    for i in range(0, len(vet_atributo)):
        for j in range(0,4):
            mat[i][4] = mat[i][4] + (vet_atributo[i][j] - valor_tirado_att[0][j])**2  
        mat[i][4] = math.sqrt(mat[i][4])
        

    return mat

def vizinhos_prox(q_vizinhos, mat_distancia):
    #verifica os k vizinhos mais proximos e retorna o indice das amostras
    #que sao mais proximas
    dist = list()
    for i in range(0,len(mat_distancia)):
        dist.append(mat_distancia[i][4])
    
    aux1 = sorted(dist)
    aux2 = list()
    
    for i in range(0, q_vizinhos):
        aux2.append(aux1[i])

    indices = list()

    for i in range(0,len(mat_distancia)):
        for j in range(0, len(aux2)):
            if mat_distancia[i][4] == aux2[j]:
                indices.append(i)

    return indices

def classifica(q_vizinhos,vet_indice, mat_respostas ):
    #pega as k amostras proxima ver qual a que tem menos distancia e 
    #retorna a classe da amostra mais proxima
    
    cont1 = 0
    cont2 = 0
    cont3 = 0
    for i in range(0, q_vizinhos):
        for j in range(0,3):
            if mat_respostas[vet_indice[i]][j] == 1:
                if j == 0:
                    cont1 += 1
                if j == 1:
                    cont2 += 1
                if j == 2:
                    cont3 += 1        
    
    aux = [cont1,cont2,cont3]
    v = max(aux)
    for i in range(0,3):
        if aux[i] == v:
            if i == 0:
                return [1,0,0]
            if i == 1:
                return [0,1,0]
            if i == 2:
                return [0,0,1]

def verifica_class(vet_classificado, vet_resposta):
    #dado o vetor resposta original da classe e a resposta produzida pelo knn 
    #é comparado se o algoritmo acertou ou nao a classe da amostra
    
    global cont
    aux = 0
    for i in range(0,3):
        a = vet_classificado[i]
        b = vet_resposta[0][i]
        if  a == b:
            aux+=1
    if aux == 3:
        print("Acertou!")
        cont +=1
    else:    
        print("Errou!")

while x < 150:
    att = cria_mat_atributos(vet_atributos)

    resp = cria_mat_resposta(vet_respostas)

    mat_dist = calcula_dist(att, valor_tirado_att)

    indices = vizinhos_prox(vizinhos, mat_dist)

    vet_classificado = classifica(vizinhos, indices,resp)
    
    verifica_class(vet_classificado, valor_tirado_resp)

    x += 1

print("Acuracia " + str((cont/150)*100) + "% " +"Quant. de amostras acertadas: " +str(cont))  