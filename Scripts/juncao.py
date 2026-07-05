import pandas as pd
import os

caminho_pasta = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv"

ficheiros = [
    ("twitter.test-majority.csv", "Twitter"),
    ("youtube.test-majority.csv", "YouTube"),
    ("hatecheck_pt_limpo.csv", "HateCheck"),
    ("toxigen_pt_limpo.csv", "ToxiGen"),
    ("dados_neutros_pt.csv", "Sinteticos_Neutros")
]

colunas_analise = [
    'D. O. Geral', 
    'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

dataframes_processados = []

print(f"A unir os {len(ficheiros)} ficheiros...\n")

for nome_ficheiro, plataforma in ficheiros:
    caminho_completo = os.path.join(caminho_pasta, nome_ficheiro)
    
    try:
        df = pd.read_csv(caminho_completo)
        df.columns = df.columns.str.strip() 

        # Lógica para detetar o nome correto da coluna de texto
        if plataforma == 'Twitter':
            coluna_texto = 'text'
        elif plataforma == 'YouTube':
            coluna_texto = 'Comentário'
        else:
            coluna_texto = 'texto'

        colunas_a_extrair = [coluna_texto] + colunas_analise
        df_filtrado = df[colunas_a_extrair].copy()

        # Uniformizar para 'texto'
        df_filtrado = df_filtrado.rename(columns={coluna_texto: 'texto'})
        
        df_filtrado['plataforma'] = plataforma
        df_filtrado['dataset_origem'] = nome_ficheiro 
        
        dataframes_processados.append(df_filtrado)
        print(f" Processado: {nome_ficheiro} ({len(df_filtrado)} linhas)")
        
    except KeyError as e:
        print(f" Erro no ficheiro {nome_ficheiro}: A coluna {e} não foi encontrada.")
    except Exception as e:
        print(f" Erro inesperado em {nome_ficheiro}: {e}")

df_final = pd.concat(dataframes_processados, ignore_index=True)

# Remover duplicados para evitar Data Leakage
df_final = df_final.drop_duplicates(subset=['texto'], keep='first')

ordem_final = ['texto', 'plataforma', 'dataset_origem'] + colunas_analise
df_final = df_final[ordem_final]

caminho_final = os.path.join(caminho_pasta, "dataset_junto_7_classes.csv")
df_final.to_csv(caminho_final, index=False, encoding='utf-8')

print("\n" + "="*50)
print("Junção do dataset feita.")
print(f"Total de frases únicas: {df_final.shape[0]}")
print(f"Categorias mantidas: {colunas_analise}")
print("="*50)
