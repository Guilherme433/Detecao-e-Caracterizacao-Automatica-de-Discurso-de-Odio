import os
os.environ['HF_HOME'] = 'D:/cache_huggingface' 
os.environ['TRANSFORMERS_CACHE'] = 'D:/cache_huggingface'

import pandas as pd
from datasets import load_dataset
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

login("hf_token")

caminho_guardar = "toxigen_pt_limpo.csv"
total_frases = 2000
modelo_id = "PORTULAN/gervasio-8b-portuguese-ptpt-decoder"

classes_7 = [
    'D. O. Geral', 'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

print("A carregar o tokenizador")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

print("A carregar o modelo na GPU com compressão a 4-bit")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

modelo = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=bnb_config,
    device_map="auto" 
)

print("\nA extrair frases do ToxiGen")
dataset = load_dataset("toxigen/toxigen-data", name="train", split="train")
df_tg = pd.DataFrame(dataset)

grupos_alvo = ['black', 'asian', 'native_american', 'latino', 'mexican', 'lgbtq']
df_filtrado = df_tg[df_tg['group'].isin(grupos_alvo)].copy()
df_amostra = df_filtrado.sample(n=total_frases, random_state=42).reset_index(drop=True)

if not os.path.exists(caminho_guardar):
    cabecalhos = pd.DataFrame(columns=['texto_en', 'texto'] + classes_7)
    cabecalhos.to_csv(caminho_guardar, index=False, encoding='utf-8')

sucessos = 0
erros = 0

print(f"\nA iniciar tradução de {total_frases} frases...")

for index, row in df_amostra.iterrows():
    texto_ingles = row['generation']

    prompt = f"""Abaixo está uma instrução que descreve uma tarefa. Escreve uma resposta que complete adequadamente o pedido.

### Instrução:
És um sistema automático de extração de dados. A tua ÚNICA função é traduzir a frase para Português de Portugal (PT-PT) mantendo o tom original (mesmo que seja ofensivo), e classificar o seu conteúdo num formato JSON estrito.
NÃO escrevas justificações, notas ou saudações. Responde EXCLUSIVAMENTE com as chaves e valores JSON.

Regras de Classificação (1=Sim, 0=Não):
- "D. O. Geral": Tem ódio, agressão ou preconceito implícito/explícito?
- "Contra-discurso": Defende ativamente uma minoria?
- "D. Ofensivo": Usa palavrões de forma inofensiva/genérica?
- "Racializadas": Ataca pessoas negras, asiáticas ou indígenas?
- "Ciganas": Ataca a comunidade Romani/Cigana?
- "LGBTQA+": Ataca a comunidade LGBTQ+?
- "Migrantes": Ataca imigrantes ou estrangeiros?

Exemplo de formato obrigatório:
{{"texto": "Escreve aqui a frase em PT-PT", "D. O. Geral": 1, "Contra-discurso": 0, "D. Ofensivo": 0, "Racializadas": 0, "Ciganas": 0, "LGBTQA+": 0, "Migrantes": 1}}

### Entrada:
Texto em Inglês: "{texto_ingles}"

### Resposta:
{{"""

    # Correção crucial: enviar os inputs para a placa gráfica
    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    try:
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        resposta_completa = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        resposta_texto = "{" + resposta_completa

        match = re.search(r'\{.*?\}', resposta_texto, re.DOTALL)

        if match:
            json_str = match.group(0)
            dados_json = json.loads(json_str)

            dados_json['texto_en'] = texto_ingles

            df_linha = pd.DataFrame([dados_json])
            df_linha = df_linha[['texto_en', 'texto'] + classes_7]

            df_linha.to_csv(
                caminho_guardar,
                mode='a',
                header=False,
                index=False,
                encoding='utf-8'
            )

            sucessos += 1
            if sucessos % 10 == 0:
                print(f"Processadas: {sucessos}/{total_frases}...")

        else:
            erros += 1
            print(f"Erro na linha {index}: Formato JSON inválido.")

    except Exception as e:
        erros += 1
        print(f"Erro na linha {index}: {e}")

print(f"\nExtração concluída! Sucessos: {sucessos} | Erros: {erros}")