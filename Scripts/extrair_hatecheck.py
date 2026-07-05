import pandas as pd
from datasets import load_dataset

dataset = load_dataset("Paul/hatecheck-portuguese", split="test")
df_hc = pd.DataFrame(dataset)

classes_7 = [
    'D. O. Geral', 'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

for col in classes_7:
    df_hc[col] = 0

df_hc.loc[df_hc['label_gold'] == 'hateful', 'D. O. Geral'] = 1

df_hc.loc[df_hc['target_ident'] == 'black people', 'Racializadas'] = 1
df_hc.loc[df_hc['target_ident'].isin(['gay people', 'trans people']), 'LGBTQA+'] = 1
df_hc.loc[df_hc['target_ident'] == 'immigrants', 'Migrantes'] = 1

if 'case_id' in df_hc.columns:
    df_hc.loc[df_hc['case_id'].str.contains('counter', case=False, na=False), 'Contra-discurso'] = 1
    
    df_hc.loc[(df_hc['case_id'].str.contains('profan|slur', case=False, na=False, regex=True)) & 
              (df_hc['label_gold'] == 'non-hateful'), 'D. Ofensivo'] = 1

df_hc = df_hc.rename(columns={'test_case': 'texto'})

colunas_finais = ['texto'] + classes_7
df_hc_limpo = df_hc[colunas_finais].copy()
df_hc_limpo = df_hc_limpo.dropna(subset=['texto'])

caminho_guardar = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\hatecheck_pt_limpo.csv"
df_hc_limpo.to_csv(caminho_guardar, index=False, encoding='utf-8')