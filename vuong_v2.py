# GLM - MODELOS PARA DADOS DE CONTAGEM
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários

# !pip install statsmodels==0.14.0

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')


# In[ ]:
##############################################################################
#                      REGRESSÃO PARA DADOS DE CONTAGEM                      #
#                 CARREGAMENTO DA BASE DE DADOS 'corruption'                 #
##############################################################################

#Fisman, R.; Miguel, E. Corruption, Norms, and Legal Enforcement:
#Evidence from Diplomatic Parking Tickets.
#Journal of Political Economy, v. 15, n. 6, p. 1020-1048, 2007.
#https://www.journals.uchicago.edu/doi/abs/10.1086/527495

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
df_corruption

#Caracterí­sticas das variáveis do dataset
df_corruption.info()

#Estatí­sticas univariadas
df_corruption.describe()


# In[ ]: Histograma da variável dependente 'violations'

plt.figure(figsize=(15,10))
sns.histplot(data=df_corruption, x='violations', bins=20, color='darkorchid')
plt.xlabel('Number of parking violations', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.show()


# In[ ]: Diagnóstico preliminar para observação de eventual igualdade entre a média e a variância da variável dependente 'violations'

print(pd.DataFrame({'Média':[df_corruption.violations.mean()],
              'Variância':[df_corruption.violations.var()]}))


# In[ ]: Comportamento das variáveis 'corruption' e 'violations' antes e
#depois do início da vigência da lei

fig, axs = plt.subplots(ncols=2, figsize=(20,10), sharey=True)

fig.suptitle('Diferença das violações de trânsito em NY antes e depois da vigência da lei',
             fontsize = 20)

post = ['no','yes']

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y']+.1, str(point['val']))

for i, v in enumerate(post):
    df = df_corruption[df_corruption.post==v]
    df['violations'] = np.log(df.violations)
    df.loc[df['violations']==np.inf, 'violations'] = 0
    df.loc[df['violations']==-np.inf, 'violations'] = 0
    sns.regplot(data=df, x='corruption', y='violations',order=3, ax=axs[i],
                color='darkorchid')
    axs[i].set_title(v)
    axs[i].set_ylabel("Violações de Trânsito em NY (logs)", fontsize = 17)
    axs[i].set_xlabel("Índice de corrupção dos paí­ses", fontsize = 17)
    label_point(df.corruption, df.violations, df.code, axs[i])  

plt.show()


# In[ ]: Estimação do modelo Poisson

y = df_corruption['violations']

x = df_corruption[['staff','post','corruption']]
X = sm.add_constant(x)
X = pd.get_dummies(X, columns=['post'], drop_first=True, dtype='int')

from statsmodels.discrete.discrete_model import Poisson

modelo_poisson = Poisson(endog=y, exog=X).fit()

modelo_poisson.summary()


# In[ ]:
##############################################################################
#           TESTE DE SUPERDISPERSÃO DE CAMERON E TRIVEDI (1990)              #
##############################################################################
#CAMERON, A. C.; TRIVEDI, P. K. Regression-based tests for overdispersion in
#the Poisson model. Journal of Econometrics, v. 46, n. 3, p. 347-364, 1990.

# Adding the Poisson model fitted values ​​(lambda_poisson) to the dataframe:
df_corruption['lambda_poisson'] = modelo_poisson.fittedvalues
df_corruption

# Creating the new Y variable*:
df_corruption['ystar'] = (((df_corruption['violations']
                            -df_corruption['lambda_poisson'])**2)
                          -df_corruption['violations'])/df_corruption['lambda_poisson']
df_corruption

# Estimating the OLS auxiliary model, without the intercept:
modelo_auxiliar = smf.ols(formula='ystar ~ 0 + lambda_poisson',
                          data=df_corruption).fit()

# Parameters of 'auxiliary_model':
modelo_auxiliar.summary()


# In[ ]:
##############################################################################
#                   ESTIMAÇÃO DO MODELO BINOMIAL NEGATIVO                    #
##############################################################################

y = df_corruption['violations']

x = df_corruption[['staff','post','corruption']]
X = sm.add_constant(x)
X = pd.get_dummies(X, columns=['post'], drop_first=True, dtype='int')

from statsmodels.discrete.discrete_model import NegativeBinomial

modelo_bneg = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

modelo_bneg.summary()


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson, modelo_bneg])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson e
#binomial negativo

# Definition of the dataframe with the models and their LL
df_llf = pd.DataFrame({'modelo':['Poisson','NB'],
                      'loglik':[modelo_poisson.llf, modelo_bneg.llf]})
df_llf

# Plot 
fig, ax = plt.subplots(figsize=(15,10))

c = ['#440154FF', '#22A884FF']

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimation", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)


# In[ ]: Adicionando os fitted values dos modelos estimados até o momento,
#para fins de comparação

df_corruption['fitted_poisson'] = np.exp(modelo_poisson.fittedvalues)
df_corruption['fitted_bneg'] = np.exp(modelo_bneg.fittedvalues)


# In[ ]: Fitted values dos modelos Poisson e binomial negativo, considerando,
#para fins didáticos, apenas a variável preditora 'staff'

plt.figure(figsize=(20,10))
sns.relplot(data=df_corruption, x='staff', y='violations',
            color='black', height=8)
sns.regplot(data=df_corruption, x='staff', y='fitted_poisson', order=3,
            color='#440154FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_bneg', order=3,
            color='#22A884FF')
plt.xlabel('Number of Diplomats (staff)', fontsize=17)
plt.ylabel('Unpaid Parking Violations (violations)', fontsize=17)
plt.legend(['Observed', 'Poisson', 'Fit Poisson', 'CI Poisson',
            'NB', 'Fit NB', 'CI BNeg'],
           fontsize=17)
plt.show


# In[ ]:
##############################################################################
#              ESTIMAÇÃO DO MODELO ZERO-INFLATED POISSON (ZIP)               #
##############################################################################

#Definição da variável dependente
y = df_corruption['violations']

#Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_corruption[['staff','post','corruption']]
X1 = sm.add_constant(x1)
X1 = pd.get_dummies(X1, columns=['post'], drop_first=True)

#Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_corruption[['corruption']]
X2 = sm.add_constant(x2)

#Estimação do modelo ZIP
#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2,
                                    inflation='logit').fit()

#Parâmetros do modelo
modelo_zip.summary()

# In[]: DEFINIÇÃO DA FUNÇÃO 'vuong_test' PARA A ELABORAÇÃO DO TESTE DE VUONG:

def vuong_test(m1, m2):
    
    from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
    from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
    from scipy.stats import norm
    
    supported_models = [ZeroInflatedPoisson,ZeroInflatedNegativeBinomialP,Poisson,NegativeBinomial]
    
    if type(m1.model) not in supported_models:
        raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
        
    if type(m2.model) not in supported_models:
        raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
    
    m1_y = m1.model.endog
    m2_y = m2.model.endog

    m1_n = len(m1_y)
    m2_n = len(m2_y)

    if m1_n == 0 or m2_n == 0:
        raise ValueError("Could not extract dependent variables from models.")

    if m1_n != m2_n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         f"Model 1 has {m1_n} observations.\n"
                         f"Model 2 has {m2_n} observations.")

    if np.any(m1_y != m2_y):
        raise ValueError("Models appear to have different values on dependent variables.")
        
    m1_linpred = pd.DataFrame(m1.predict(which="prob"))
    m2_linpred = pd.DataFrame(m2.predict(which="prob"))        

    m1_probs = np.repeat(np.nan, m1_n)
    m2_probs = np.repeat(np.nan, m2_n)

    which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(m1_linpred.columns) else None for x in m1_y]    
    which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(m2_linpred.columns) else None for x in m2_y]

    for i, v in enumerate(m1_probs):
        m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

    for i, v in enumerate(m2_probs):
        m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

    lm1p = np.log(m1_probs)
    lm2p = np.log(m2_probs)

    m = lm1p - lm2p

    v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

    pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

    print("Vuong Non-Nested Hypothesis Test-Statistic (Raw):")
    print(f"Vuong z-statistic: {v}")
    print(f"p-value: {pval}")

# In[]: TESTE DE VUONG: POISSON X ZIP

vuong_test(modelo_poisson, modelo_zip)

# In[ ]: Gráfico para comparar os valores previstos x valores reais de
#'violations' pelo modelo ZIP

zip_predictions = modelo_zip.predict(X1, exog_infl=X2)
predicted_counts = np.round(zip_predictions)
actual_counts = df_corruption['violations']

plt.figure(figsize=(15,10))
plt.plot(df_corruption.index, predicted_counts, 'go-',
         color='orange')
plt.plot(df_corruption.index, actual_counts, 'go-',
         color='#440154FF')
plt.xlabel('Observation', fontsize=20)
plt.ylabel('Traffic violations', fontsize=20)
plt.legend(['Predicted values with ZIP', 'Observed values from the Dataset'],
           fontsize=20)
plt.show()


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_poisson, modelo_zip])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson, BNeg e ZIP

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','BNeg'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ["#440154FF", "#453781FF", "#22A884FF"]

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimação", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)


# In[ ]:
##############################################################################
#        ESTIMAÇÃO DO MODELO ZERO-INFLATED BINOMIAL NEGATIVO (ZINB)          #
##############################################################################

#Definição da variável dependente
y = df_corruption['violations']

#Definição das variáveis preditoras que entrarão no componente de contagem
x1 = df_corruption[['staff','post','corruption']]
X1 = sm.add_constant(x1)
X1 = pd.get_dummies(X1, columns=['post'], drop_first=True)

#Definição das variáveis preditoras que entrarão no componente logit (inflate)
x2 = df_corruption[['corruption']]
X2 = sm.add_constant(x2)

#Estimação do modelo ZINB

from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

#O argumento 'exog_infl' corresponde às variáveis que entram no componente
#logit (inflate)
modelo_zinb = ZeroInflatedNegativeBinomialP(y, X1, exog_infl=X2,
                                            inflation='logit').fit()

#Parâmetros do modelo
modelo_zinb.summary()


# In[]: TESTE DE VUONG: BNEG X ZINB

vuong_test(modelo_bneg,modelo_zinb)

# In[ ]: Gráfico para comparar os valores previstos x valores reais de
#'violations' pelo modelo ZINB

zinb_predictions = modelo_zinb.predict(X1, exog_infl=X2)
predicted_counts = np.round(zinb_predictions)
actual_counts = df_corruption['violations']

plt.figure(figsize=(15,10))
plt.plot(df_corruption.index, predicted_counts, 'go-',
         color='orange')
plt.plot(df_corruption.index, actual_counts, 'go-',
         color='#440154FF')
plt.xlabel('ObservaÇÃO', fontsize=20)
plt.ylabel('Violações de Trânsito', fontsize=20)
plt.legend(['Valores Previstos pelo ZINB', 'Valores Reais no Dataset'],
           fontsize=20)
plt.show()


# In[ ]: likelihood ratio test para comparação de LL's entre modelos

#Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llnull
    llk_2 = modelo_1.llf
    
    if len(modelos)>1:
        llk_1 = modelo_1.llf
        llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest([modelo_bneg, modelo_zinb])


# In[ ]: Gráfico para a comparação dos LL dos modelos Poisson, BNeg, ZIP e
#ZINB

#Definição do dataframe com os modelos e respectivos LL
df_llf = pd.DataFrame({'modelo':['Poisson','ZIP','NB','ZINB'],
                      'loglik':[modelo_poisson.llf,
                                modelo_zip.llf,
                                modelo_bneg.llf,
                                modelo_zinb.llf]})
df_llf

#Plotagem propriamente dita
fig, ax = plt.subplots(figsize=(15,10))

c = ["#440154FF", "#453781FF", "#22A884FF", "orange"]

ax1 = ax.barh(df_llf.modelo, df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=24)
ax.set_ylabel("Estimation", fontsize=20)
ax.set_xlabel("Log-Likehood", fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)


# In[ ]: Adicionando os fitted values dos modelos estimados para fins de
#comparação

df_corruption['fitted_zip'] = modelo_zip.predict(X1, exog_infl=X2)
df_corruption['fitted_zinb'] = modelo_zinb.predict(X1, exog_infl=X2)
df_corruption


# In[ ]: Fitted values dos modelos POISSON, BNEG, ZIP e ZINB, considerando,
#para fins didáticos, a variável dependente 'violations' em função apenas da
#variável preditora 'staff'

plt.figure(figsize=(20,10))
sns.relplot(data=df_corruption, x='staff', y='violations',
            color='black', height=8)
sns.regplot(data=df_corruption, x='staff', y='fitted_poisson', order=3,
            color='#440154FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_bneg', order=3,
            color='#22A884FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_zip', order=3,
            color='#453781FF')
sns.regplot(data=df_corruption, x='staff', y='fitted_zinb', order=3,
            color='orange')
plt.xlabel('Number of Diplomats (staff)', fontsize=17)
plt.ylabel('Unpaid Parking Violations (violations)', fontsize=17)
plt.legend(['Observed', 'Poisson', 'Fit Poisson', 'CI Poisson',
            'NB', 'Fit NB', 'CI NB',
            'ZIP', 'Fit ZIP', 'CI ZIP',
            'ZINB', 'Fit ZINB', 'CI ZINB'],
           fontsize=14)
plt.show

################################## FIM ######################################