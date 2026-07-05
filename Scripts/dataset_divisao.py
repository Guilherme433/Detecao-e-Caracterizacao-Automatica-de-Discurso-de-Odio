import pandas as pd
from sklearn.model_selection import train_test_split

caminho_csv = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_junto_7_classes.csv"

classes = [
    'D. O. Geral', 'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

print("A carregar o dataset")
df = pd.read_csv(caminho_csv)

df = df.dropna(subset=['texto'] + classes)

# divisao 70/15/15
df_resto, df_teste = train_test_split(df, test_size=0.15, random_state=42)
df_treino, df_val = train_test_split(df_resto, test_size=0.1764, random_state=42)

caminho_teste = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_teste_oficial.csv"
caminho_val = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_validacao_oficial.csv"
caminho_treino_base = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_treino_base.csv"

df_teste.to_csv(caminho_teste, index=False, encoding='utf-8')
df_val.to_csv(caminho_val, index=False, encoding='utf-8')
df_treino.to_csv(caminho_treino_base, index=False, encoding='utf-8')

print("\n" + "="*50)
print("Divisão Concluída")
print("="*50)
print(f"Conjunto Treino: {len(df_treino)} frases guardadas em 'dataset_treino_base.csv'")
print(f"Conjunto Validação: {len(df_val)} frases guardadas em 'dataset_validacao_oficial.csv'")
print(f"Conjunto Teste): {len(df_teste)} frases guardadas em 'dataset_teste_oficial.csv'")
print("="*50)