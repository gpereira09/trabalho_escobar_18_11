import pandas as pd
from pickle import load

nova_paciente = {
    'age': '40-49',
    'menopause': 'premeno',
    'tumor-size': '30-34',
    'inv-nodes': '3-5',
    'node-caps': 'yes',
    'deg-malig': 3,
    'breast': 'left',
    'breast-quad': 'left_low',
    'irradiat': 'yes'
}

normalizador = load(open('normalizador_breast_cancer.model', 'rb'))
colunas = load(open('colunas_breast_cancer.model', 'rb'))
modelo = load(open('breast_cancer_modelo.model', 'rb'))

colunas_atributos = colunas['colunas_atributos']
colunas_numericas = colunas['colunas_numericas']
colunas_categoricas = colunas['colunas_categoricas']
colunas_dummies = colunas['colunas_dummies']
colunas_modelo = colunas['colunas_modelo']
modas = colunas['modas']


def faixa_para_numero(valor):
    valor = str(valor)
    if '-' in valor:
        inicio, fim = valor.split('-')
        return (float(inicio) + float(fim)) / 2
    return float(valor)


nova_paciente = pd.DataFrame([nova_paciente])[colunas_atributos]

# Valores ausentes ('?' ou vazio) recebem a moda da base de treinamento
nova_paciente = nova_paciente.replace('?', pd.NA).fillna(modas)

nova_numericos = pd.DataFrame(
    normalizador.transform(nova_paciente[colunas_numericas].apply(lambda c: c.map(faixa_para_numero))),
    columns=colunas_numericas
)
nova_categoricos = pd.get_dummies(nova_paciente[colunas_categoricas].astype(str), prefix_sep='_', dtype=int)
nova_categoricos = nova_categoricos.reindex(columns=colunas_dummies, fill_value=0)

nova_normalizada = nova_numericos.join(nova_categoricos)[colunas_modelo]

diagnostico = modelo.predict(nova_normalizada)[0]

print('Modelo utilizado:', colunas['nome_modelo'])
print('Dados da nova paciente:')
print(nova_paciente.iloc[0].to_string())
print('\nIndicativo de diagnóstico (Class):', diagnostico)

if hasattr(modelo, 'predict_proba'):
    probabilidades = modelo.predict_proba(nova_normalizada)[0]
    for classe, probabilidade in zip(modelo.classes_, probabilidades):
        print(f'  P({classe}) = {probabilidade:.2%}')
