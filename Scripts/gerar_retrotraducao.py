import pandas as pd
from deep_translator import GoogleTranslator

caminho_treino_base = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_treino_base.csv"
caminho_treino_aumentado = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_treino_retrotraducao.csv"

df_treino = pd.read_csv(caminho_treino_base)

# Só queremos gerar dados sintéticos para frases que contêm ódio
df_odio = df_treino[df_treino['D. O. Geral'] == 1.0].copy()

print(f"Quantidade frases treino:{len(df_treino)}")
print(f"Quantidade frases treino:{len(df_odio)}")
print("\nA iniciar a Retrotradução (PT -> EN -> PT)")

tradutor_pt_en = GoogleTranslator(source='pt', target='en')
tradutor_en_pt = GoogleTranslator(source='en', target='pt')

frases_sinteticas = []

for index, row in df_odio.iterrows():
    texto_original = row['texto']
    
    try:
        texto_en = tradutor_pt_en.translate(texto_original)
        texto_pt_sintetico = tradutor_en_pt.translate(texto_en)
        
        nova_linha = row.copy()
        nova_linha['texto'] = texto_pt_sintetico
        nova_linha['dataset_origem'] = 'sintetico_retrotraducao' 
        
        frases_sinteticas.append(nova_linha)
        
        if len(frases_sinteticas) % 50 == 0:
            print(f"Geradas {len(frases_sinteticas)} frases...")
            print(f"  Original : {texto_original}")
            print(f"  Sintética: {texto_pt_sintetico}\n")
            
    except Exception as e:
        continue


df_sintetico = pd.DataFrame(frases_sinteticas)


df_treino_final = pd.concat([df_treino, df_sintetico], ignore_index=True)
df_treino_final = df_treino_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_treino_final.to_csv(caminho_treino_aumentado, index=False, encoding='utf-8')

print("\n" + "="*50)
print("="*50)
print(f"Tamanho do Treino Base Original: {len(df_treino)}")
print(f"Tamanho do Novo Treino (Base + Sintéticos): {len(df_treino_final)}")
print(f"Ficheiro guardado em: {caminho_treino_aumentado}")
print("="*50)