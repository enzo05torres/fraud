# %%
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("card_transdata.csv")

# %%
df.head(5)

# %%
df.shape

# %%
df.describe()

# %%
df.info()

# %%
(df["fraud"].sum() / len(df))

# %%
df_corr = df.corr()
df_corr

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.show()

# %%
fraude_counts = df['fraud'].value_counts()

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=fraude_counts.index, y=fraude_counts.values, palette='viridis')
plt.title('Fraudes e Não Fraudes')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.ylabel('Número de Transações')
plt.legend(bars.patches, ['Não Fraude (0)', 'Fraude (1)'])
for bar in bars.patches:
    bars.annotate(format(int(bar.get_height())), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='bottom', fontsize=12)
plt.show()

# %%
chip_fraude_counts = df.groupby(['used_chip', 'fraud']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
bars = chip_fraude_counts.plot(kind='bar', color=['lightblue', 'salmon'], edgecolor='black')
plt.title('Comparação de Fraudes e Não Fraudes com Base no Uso do Chip')
plt.xlabel('Uso do Chip (0 = Não, 1 = Sim)')
plt.ylabel('Número de Transações')

plt.legend(['Não Fraude (0)', 'Fraude (1)'])

for bar in bars.patches:
    bars.annotate(format(int(bar.get_height())), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='bottom', fontsize=12)
plt.grid(axis='y')
plt.xticks(rotation=0) 
plt.show()

# %%
chip_fraude_counts = df.groupby(['used_chip', 'fraud']).size().unstack(fill_value=0)

total_com_chip = chip_fraude_counts.loc[1].sum()
total_sem_chip = chip_fraude_counts.loc[0].sum()

fraudes_com_chip = chip_fraude_counts.loc[1, 1]
fraudes_sem_chip = chip_fraude_counts.loc[0, 1]

porcentagem_fraudes_com_chip = (fraudes_com_chip / total_com_chip) * 100
porcentagem_fraudes_sem_chip = (fraudes_sem_chip / total_sem_chip) * 100

print(f"Porcentagem de fraudes com uso de chip: {porcentagem_fraudes_com_chip:.2f}%")
print(f"Porcentagem de fraudes sem uso de chip: {porcentagem_fraudes_sem_chip:.2f}%")

# %%
online_counts = df.groupby(['online_order', 'fraud']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
bars = online_counts.plot(kind='bar', color=['lightblue', 'salmon'], edgecolor='black')
plt.title('Comparação de Fraudes e Não Fraudes com Base em Compras Online')
plt.xlabel('Compra Online (0 = Não, 1 = Sim)')
plt.ylabel('Número de Transações')
plt.legend(['Não Fraude (0)', 'Fraude (1)'])

for bar in bars.patches:
    bars.annotate(format(int(bar.get_height())), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='bottom', fontsize=12)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.show()

# %%
online_fraude_counts = df.groupby(['online_order', 'fraud']).size().unstack(fill_value=0)

total_online = online_fraude_counts.loc[1].sum() 
total_nao_online = online_fraude_counts.loc[0].sum() 

fraudes_online = online_fraude_counts.loc[1, 1]
fraudes_nao_online = online_fraude_counts.loc[0, 1] 

porcentagem_fraudes_online = (fraudes_online / total_online) * 100
porcentagem_fraudes_nao_online = (fraudes_nao_online / total_nao_online) * 100

print(f"Porcentagem de fraudes em compras online: {porcentagem_fraudes_online:.2f}%")
print(f"Porcentagem de fraudes em compras não online: {porcentagem_fraudes_nao_online:.2f}%")

# %%
df['fraud'] = df['fraud'].astype('category')

plt.figure(figsize=(12, 8))
sns.stripplot(y='distance_from_home', x='fraud', 
               data=df, 
               hue='fraud', 
               palette={0: 'blue', 1: 'red'}, 
               alpha=0.7, 
               dodge=True, 
               jitter=True) 

plt.title('Relação entre a Distância da casa e Fraude')
plt.ylabel('Distância de casa')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.xticks([0, 1], ['Não Fraude', 'Fraude']) 
plt.grid(True)
plt.show()

# %%
df['fraud'] = df['fraud'].astype('category')

plt.figure(figsize=(12, 8))
sns.stripplot(y='distance_from_last_transaction', x='fraud', 
               data=df, 
               hue='fraud', 
               palette={0: 'blue', 1: 'red'}, 
               alpha=0.7, 
               dodge=True, 
               jitter=True) 

plt.title('Relação entre a Distância da Última Transação e Fraude')
plt.ylabel('Distância da Última Transação')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.xticks([0, 1], ['Não Fraude', 'Fraude']) 
plt.grid(True)
plt.show()

# %%
usedpin_counts = df.groupby(['used_pin_number', 'fraud']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
bars = usedpin_counts.plot(kind='bar', color=['lightblue', 'salmon'], edgecolor='black')
plt.title('Comparação de Fraudes e Não Fraudes com Base no Uso da senha')
plt.xlabel('Uso da senha (0 = Não, 1 = Sim)')
plt.ylabel('Número de Transações')

plt.legend(['Não Fraude (0)', 'Fraude (1)'])

for bar in bars.patches:
    bars.annotate(format(int(bar.get_height())), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=0) 
plt.show()

# %%
usedpin_fraud_counts = df.groupby(['used_pin_number', 'fraud']).size().unstack(fill_value=0)

total_usedpin = usedpin_fraud_counts.loc[1].sum() 
total_nao_usedpin = usedpin_fraud_counts.loc[0].sum() 

fraudes_usedpin = usedpin_fraud_counts.loc[1, 1]
fraudes_nao_usedpin = usedpin_fraud_counts.loc[0, 1] 

porcentagem_usedpin = (fraudes_usedpin / total_usedpin) * 100
porcentagem_nao_usedpin = (fraudes_nao_usedpin / total_nao_usedpin) * 100

print(f"Porcentagem de fraudes em compras usando a senha: {porcentagem_usedpin:.2f}%")
print(f"Porcentagem de fraudes em compras não usando a senha: {porcentagem_nao_usedpin:.2f}%")

# %%
rpt_counts = df.groupby(['repeat_retailer', 'fraud']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
bars = chip_fraude_counts.plot(kind='bar', color=['lightblue', 'salmon'], edgecolor='black')
plt.title('Comparação de Fraudes e Não Fraudes com Base no cliente')
plt.xlabel('Uso do Chip (0 = Não, 1 = Sim)')
plt.ylabel('Número de Transações')
plt.legend(['Não Fraude (0)', 'Fraude (1)'])

for bar in bars.patches:
    bars.annotate(format(int(bar.get_height())), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='bottom', fontsize=12)
plt.grid(axis='y')
plt.xticks(rotation=0) 
plt.show()

# %%
rpt_fraud_counts = df.groupby(['repeat_retailer', 'fraud']).size().unstack(fill_value=0)

total_rpt = rpt_fraud_counts.loc[1].sum() 
total_nao_rpt = rpt_fraud_counts.loc[0].sum() 

fraudes_rpt = rpt_fraud_counts.loc[1, 1]
fraudes_nao_rpt = rpt_fraud_counts.loc[0, 1] 

porcentagem_rpt = (fraudes_rpt / total_rpt) * 100
porcentagem_nao_rpt = (fraudes_nao_rpt / total_nao_rpt) * 100

print(f"Porcentagem de fraudes em compras de clientes sempre: {porcentagem_rpt:.2f}%")
print(f"Porcentagem de fraudes em compras de nao clientes: {porcentagem_nao_rpt:.2f}%")

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='fraud', y='ratio_to_median_purchase_price', data=df)

plt.title('Proporção do Preço da Transação em Relação à Fraude')
plt.xlabel('Fraude (0 = Não, 1 = Sim)')
plt.ylabel('Proporção entre Preço da Transação e Preço Médio')
plt.xticks([0, 1], ['Não Fraude', 'Fraude'])
plt.grid(True)
plt.show()

# %%


# %%
X = df.drop(columns=["fraud"])
y = df["fraud"]

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print(classification_report(y_test, y_pred))

# %%
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.xticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.yticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

#knn_model.fit(X_train, y_train)

#y_pred = knn_model.predict(X_test)

#print(classification_report(y_test, y_pred))

# %%
#cm = confusion_matrix(y_test, y_pred)

#plt.figure(figsize=(8, 6))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#plt.title('Matriz de Confusão')
#plt.xlabel('Previsão')
#plt.ylabel('Valor Real')
#plt.xticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
#plt.yticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
#plt.show()

# %%
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.model_selection import cross_val_score

# %%
param_grid = {
    'n_neighbors': Integer(1, 10),          
    'weights': Categorical(['distance']),  
    'p': Integer(1, 2)                
}

opt = BayesSearchCV(
    knn_model,
    param_grid,
    n_iter=10,  
    scoring='roc_auc', 
    cv=2,
    random_state=42
)

# %%
opt.fit(X_train, y_train)

#%%
opt.best_params_

# %%
print("Melhor Score: ", opt.best_score_)

# %%
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.xticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.yticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.show()

# %%
y_pred = opt.predict(X_test)

# %%
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.xticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.yticks(ticks=[0.5, 1.5], labels=['Não Fraude', 'Fraude'])
plt.show()

# %%
