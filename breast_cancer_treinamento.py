import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 1. Leitura dos dados
dados = pd.read_csv('breast-cancer.csv')
print('Dimensão da base:', dados.shape)
print(dados['Class'].value_counts(), '\n')

# 2. Tratamento de valores ausentes ('?') -> substitui pela moda da coluna
colunas_atributos = [c for c in dados.columns if c != 'Class']
dados = dados.replace('?', pd.NA)
print('Valores ausentes por coluna:')
print(dados.isna().sum()[dados.isna().sum() > 0], '\n')
modas = {c: dados[c].mode()[0] for c in colunas_atributos}
dados = dados.fillna(modas)

# 3. Separação de atributos e classe
atributos = dados[colunas_atributos]
classes = dados['Class']

# 4. Atributos ordinais (faixas) viram número (ponto médio da faixa); demais são categóricos
colunas_numericas = ['age', 'tumor-size', 'inv-nodes', 'deg-malig']
colunas_categoricas = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']


def faixa_para_numero(valor):
    valor = str(valor)
    if '-' in valor:
        inicio, fim = valor.split('-')
        return (float(inicio) + float(fim)) / 2
    return float(valor)


def preparar_numericos(df):
    return df[colunas_numericas].apply(lambda coluna: coluna.map(faixa_para_numero))


# 5. Divisão treino/teste (estratificada, 70/30)
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    atributos, classes, test_size=0.3, random_state=42, stratify=classes
)

# 6. Normalização (MinMax nos numéricos, dummies nos categóricos) ajustada só no treino
normalizador = MinMaxScaler()
normalizador.fit(preparar_numericos(atributos_treino))

colunas_dummies = list(pd.get_dummies(atributos[colunas_categoricas], prefix_sep='_', dtype=int).columns)


def normalizar(df, modelo_normalizador):
    numericos = pd.DataFrame(
        modelo_normalizador.transform(preparar_numericos(df)),
        columns=colunas_numericas, index=df.index
    )
    categoricos = pd.get_dummies(df[colunas_categoricas], prefix_sep='_', dtype=int)
    categoricos = categoricos.reindex(columns=colunas_dummies, fill_value=0)
    return numericos.join(categoricos)


treino_normalizado = normalizar(atributos_treino, normalizador)
teste_normalizado = normalizar(atributos_teste, normalizador)
colunas_modelo = list(treino_normalizado.columns)

# 7. Modelos candidatos e hiperparâmetros
#    O balanceamento (SMOTE) entra como hiperparâmetro: a busca decide se usar ou não,
#    sempre pelo critério de maior acurácia. O SMOTE é aplicado só nas dobras de treino.
modelos = {
    'Árvore de Decisão': (DecisionTreeClassifier(random_state=42), {
        'classificador__criterion': ['gini', 'entropy'],
        'classificador__max_depth': [2, 3, 4, 5, 8, None],
        'classificador__min_samples_leaf': [1, 5, 10],
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'classificador__n_estimators': [100, 300],
        'classificador__max_depth': [3, 5, 8, None],
        'classificador__min_samples_leaf': [1, 5],
    }),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {
        'classificador__n_estimators': [50, 100, 200],
        'classificador__learning_rate': [0.01, 0.05, 0.1],
        'classificador__max_depth': [1, 2, 3],
    }),
    'KNN': (KNeighborsClassifier(), {
        'classificador__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'classificador__weights': ['uniform', 'distance'],
    }),
    'Naive Bayes': (GaussianNB(), {
        'classificador__var_smoothing': [1e-9, 1e-6, 1e-3, 1e-1],
    }),
    'SVM': (SVC(random_state=42), {
        'classificador__C': [0.1, 1, 10],
        'classificador__kernel': ['linear', 'rbf'],
        'classificador__gamma': ['scale', 0.1, 1],
    }),
    'Regressão Logística': (LogisticRegression(max_iter=2000), {
        'classificador__C': [0.01, 0.1, 1, 10],
    }),
    'Rede Neural (MLP)': (MLPClassifier(max_iter=2000, random_state=42), {
        'classificador__hidden_layer_sizes': [(5,), (10,), (10, 5)],
        'classificador__alpha': [0.0001, 0.01, 1.0],
    }),
}

validacao_cruzada = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
resultados = []
melhores_buscas = {}

for nome, (classificador, grade) in modelos.items():
    pipeline = Pipeline([('balanceador', SMOTE(random_state=42)), ('classificador', classificador)])
    grade = {'balanceador': [SMOTE(random_state=42), 'passthrough'], **grade}
    busca = GridSearchCV(pipeline, grade, scoring='accuracy', cv=validacao_cruzada, n_jobs=-1)
    busca.fit(treino_normalizado, classes_treino)
    melhores_buscas[nome] = busca

    acuracia_teste = accuracy_score(classes_teste, busca.predict(teste_normalizado))
    resultados.append({
        'modelo': nome,
        'acuracia_validacao_cruzada': round(busca.best_score_, 4),
        'desvio_validacao_cruzada': round(busca.cv_results_['std_test_score'][busca.best_index_], 4),
        'acuracia_teste': round(acuracia_teste, 4),
        'balanceamento': 'SMOTE' if busca.best_params_['balanceador'] != 'passthrough' else 'não',
        'melhores_parametros': {k.replace('classificador__', ''): v
                                for k, v in busca.best_params_.items() if k != 'balanceador'},
    })
    print(f'{nome}: acurácia CV = {busca.best_score_:.4f} | acurácia teste = {acuracia_teste:.4f}')

# 8. Comparação e escolha do modelo de melhor acurácia (validação cruzada no treino)
#    Critério: maior acurácia média na validação cruzada; empate -> menor desvio padrão
resultados = pd.DataFrame(resultados).sort_values(
    ['acuracia_validacao_cruzada', 'desvio_validacao_cruzada'], ascending=[False, True]
).reset_index(drop=True)
resultados.to_csv('resultados_modelos.csv', index=False)
print('\nRanking dos modelos:')
print(resultados[['modelo', 'acuracia_validacao_cruzada', 'desvio_validacao_cruzada', 'acuracia_teste', 'balanceamento']].to_string())

nome_melhor = resultados.loc[0, 'modelo']
melhor_busca = melhores_buscas[nome_melhor]
print(f'\nMelhor modelo: {nome_melhor}')
print('Parâmetros:', melhor_busca.best_params_)

previsoes_teste = melhor_busca.predict(teste_normalizado)
print(f'Acurácia no teste: {accuracy_score(classes_teste, previsoes_teste):.4f}')
print(classification_report(classes_teste, previsoes_teste))

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix(classes_teste, previsoes_teste, labels=melhor_busca.classes_),
    display_labels=melhor_busca.classes_
).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Matriz de confusão (teste) - {nome_melhor}')
fig.tight_layout()
fig.savefig('matriz_confusao_breast_cancer.png')

fig, ax = plt.subplots(figsize=(8, 5))
ordem = resultados.sort_values('acuracia_validacao_cruzada')
ax.barh(ordem['modelo'], ordem['acuracia_validacao_cruzada'], label='Validação cruzada')
ax.scatter(ordem['acuracia_teste'], ordem['modelo'], color='black', zorder=3, label='Teste')
ax.set(xlabel='Acurácia', title='Comparação dos modelos', xlim=(0.5, 0.85))
ax.legend(loc='lower right')
ax.grid(axis='x')
fig.tight_layout()
fig.savefig('comparacao_modelos_breast_cancer.png')

# 9. Modelo final: normalizador e melhor modelo re-treinados com a base completa
normalizador_final = MinMaxScaler().fit(preparar_numericos(atributos))
dump(normalizador_final, open('normalizador_breast_cancer.model', 'wb'))

dados_normalizados = normalizar(atributos, normalizador_final)
modelo_final = melhor_busca.best_estimator_.fit(dados_normalizados, classes)
dump(modelo_final, open('breast_cancer_modelo.model', 'wb'))

dump({
    'colunas_atributos': colunas_atributos,
    'colunas_numericas': colunas_numericas,
    'colunas_categoricas': colunas_categoricas,
    'colunas_dummies': colunas_dummies,
    'colunas_modelo': colunas_modelo,
    'modas': modas,
    'nome_modelo': nome_melhor,
}, open('colunas_breast_cancer.model', 'wb'))

print('Modelos salvos: normalizador_breast_cancer.model, colunas_breast_cancer.model, breast_cancer_modelo.model')
