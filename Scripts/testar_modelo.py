import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# O caminho para o modelo
caminho_modelo = r"D:\Projetos\Projeto_final_CURSO\Projeto\modelo_v1_7classes"

classes_7 = [
    'D. O. Geral', 'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

print("\n" + "="*50)
print("A carregar o modelo de análise de  ódio.")
print("="*50)

# Carregar o Tokenizador e o Modelo Treinado
try:
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    
    # Avaliar modelo
    modelo.eval()
    print("Modelo carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifica se o caminho da pasta está correto.")
    exit()

print("\n" + "="*50)
print("Escreve uma frase para testar. Escreve 'sair' para fechar.")
print("="*50 + "\n")

# O Ciclo Infinito de Teste
while True:
    frase_teste = input("Escreve uma frase: ")
    
    if frase_teste.strip().lower() == 'sair':
        print("A encerrar o sistema. Bom trabalho!")
        break
        
    if not frase_teste.strip():
        continue
        
    # Preparar a frase para o modelo ler
    inputs = tokenizer(
        frase_teste, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    # Fazer a previsão sem calcular gradientes 
    with torch.no_grad():
        outputs = modelo(**inputs)
        
    # Extrair os resultados
    logits = outputs.logits
    probabilidades = torch.sigmoid(logits)
    
    # Transformar probabilidades em 0 1
    previsoes = (probabilidades > 0.4).int().tolist()[0]
    
    print("\nResultados análise")
    print("-" * 40)
    
    tem_odio = False
    
    for nome_classe, valor_binario, prob in zip(classes_7, previsoes, probabilidades[0].tolist()):
        percentagem = prob * 100
        resultado_texto = "SIM" if valor_binario == 1 else "Não"
        
        
        if valor_binario == 1:
            tem_odio = True
            print(f"  ➜ {nome_classe.ljust(16)}: {resultado_texto} ({percentagem:.1f}%)")
        else:
            print(f"    {nome_classe.ljust(16)}: {resultado_texto} ({percentagem:.1f}%)")
            
    print("-" * 40)
    if not tem_odio:
        print("Frase Limpa / Inofensiva")
    print("\n")


'''
Estes migrantes vêm para o nosso país roubar os subsídios todos, deviam ser recambiados."

Teste Neutro (Deve dar Não em tudo):

"O meu vizinho é migrante e ontem ajudou-me a carregar as compras até ao prédio."

2. Categoria: LGBTQA+
 Teste de Ódio (Deve dar Sim em D.O. Geral e LGBTQA+):

"Esses gays e trans são uma vergonha para a sociedade, é uma doença."

Teste Neutro (Deve dar Não em tudo):

"Fui ao casamento de dois amigos homossexuais e a cerimónia foi lindíssima."

Categoria: Ciganas


"Os ciganos são todos um bando de ladrões que não querem trabalhar."

 Teste Neutro:

"Comprei esta manta na feira a um vendedor de etnia cigana."

Teste de Ódio:

"Voltem para a vossa terra, esta raça não pertence à Europa."

"A comunidade asiática organizou um evento cultural espetacular no centro de Lisboa."
'''