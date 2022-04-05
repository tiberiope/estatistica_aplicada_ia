import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

base = pd.read_csv('CSV/mt_cars.csv')
print(base.shape)

#mpg consumo, cyl clindros, disp cilindradas, hp
print(base.head())

#exclui coluna 1
base = base.drop(['Unnamed: 0'], axis = 1)

#Cálculo da correlação entre x (independente) e y (dependente)
X = base.iloc[:, 2].values #coluna disp
y = base.iloc[:, 0].values #coluna mpg
correlacao = np.corrcoef(X, y)
correlacao

# Mudança do formato de X para o formato de matriz (necessário para versões mais recentes do sklearn)
X = X.reshape(-1, 1)

# Criação do modelo, treinamento, visualização dos coeficientes e do score do modelo
modelo = LinearRegression()
modelo.fit(X, y)

#Interceptação
modelo.intercept_

#inclinação
modelo.coef_

#score R^2
modelo.score(X, y)

# Geração das previsões
previsoes = modelo.predict(X)
previsoes

# Criação do modelo, utilizando a biblioteca statsmodel
#podemos ver r ajustadodo r2
modelo_ajustado = sm.ols(formula = 'mpg ~ disp', data = base)
modelo_treinado = modelo_ajustado.fit()
modelo_treinado.summary()

# Visualização dos resultados
plt.scatter(X, y)
plt.plot(X, previsoes, color = 'red')

# Previsão para somente um valor
modelo.predict([[200]])

# Criação de novas variáveis X1 e Y1 e novo modelo para comparação com o anterior
# 3 variáveis dependentes para prever mpg: cyl	disp	hp
X1 = base.iloc[:, 1:4].values
X1

y1 = base.iloc[:, 0].values
modelo2 = LinearRegression()
modelo2.fit(X1, y1)
#R^2
modelo2.score(X1, y1)

# Criação do modelo ajustado com mais atributos (regressão linear múltipla)
#usando stats models
modelo_ajustado2 = sm.ols(formula = 'mpg ~ cyl + disp + hp', data = base)
modelo_treinado2 = modelo_ajustado2.fit()
modelo_treinado2.summary()

# Previsão de um novo registro
novo = np.array([4, 200, 100])
novo = novo.reshape(1, -1)
modelo2.predict(novo)
