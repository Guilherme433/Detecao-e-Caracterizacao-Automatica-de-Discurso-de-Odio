# Deteção e Caracterização Automática de Discurso de Ódio

Este repositório contém o código-fonte desenvolvido no âmbito da dissertação focada na deteção e classificação automática de Discurso de Ódio (DO) para o Português de Portugal (PT-PT). 

O sistema baseia-se na afinação (fine-tuning) do modelo fundacional Albertina PT-PT, adaptado para uma tarefa de classificação multi-label capaz de distinguir nuances semânticas complexas, como discurso ofensivo inócuo, ódio direcionado a minorias e contra-discurso.

---

## Taxonomia de Classificação

O modelo foi treinado para identificar simultaneamente a presença das seguintes 7 categorias num dado texto (limiar de ativação global otimizado para 0.4):

* D. O. Geral: Ódio, agressão ou preconceito implícito/explícito.
* Contra-discurso: Defesa ativa de uma minoria ou refutação de narrativas de ódio.
* D. Ofensivo: Uso inofensivo/genérico de calão ou palavrões.
* Racializadas: Ataques a minorias étnicas racializadas.
* Ciganas: Ataques à comunidade Romani/Cigana.
* LGBTQA+: Ataques à comunidade LGBTQA+.
* Migrantes: Ataques a imigrantes ou estrangeiros.

---

## Estrutura do Repositório e Scripts

A pipeline de dados foi construída de forma estrita e cronológica para garantir o rigor científico e prevenir fenómenos de Data Leakage. Abaixo detalha-se a função de cada script:

### 1. Curadoria de Dados
* extrair_toxigen.py
    * Função: Recorre ao modelo de linguagem (LLM) Gervásio 8B (quantizado a 4-bits) para traduzir e re-anotar o dataset inglês ToxiGen para PT-PT. 
    * Destaque Técnico: Utiliza One-Shot Prompting para forçar a saída em formato JSON estrito e implementa um mecanismo de tolerância a falhas (Fault Tolerance) com gravação contínua em disco para prevenir a perda de dados em caso de erros de hardware (OOM).

### 2. Pipeline de Processamento
* juncao.py (Fase 1)
    * Função: Unifica as diferentes fontes de dados (Twitter, YouTube, ToxiGen, HateCheck e Dados Neutros sintéticos) num único dataset tabular, normalizando as colunas para a taxonomia do projeto.
* dataset_divisao.py (Fase 2)
    * Função: Divide o dataset unificado de forma aleatória e reprodutível em subconjuntos de Treino (70%), Validação (15%) e Teste (15%). Os conjuntos de validação e teste são trancados nesta fase.
* gerar_retrotraducao.py (Fase 3)
    * Função: Aplica a técnica de Data Augmentation por retrotradução (PT -> EN -> PT) exclusivamente ao conjunto de treino. Aumenta a volumetria e a diversidade sintática dos dados sem contaminar a avaliação.

### 3. Modelação e Inferência
* treino_modelo.py
    * Função: Afina o codificador Albertina PT-PT utilizando a API Hugging Face Trainer. Implementa a função de perda Binary Cross-Entropy with Logits e um mecanismo de salvaguarda do melhor checkpoint para mitigar overfitting.
* testar_modelo.py
    * Função: Interface de linha de comandos (CLI) que permite carregar o modelo treinado e classificar novos textos em tempo real. Demonstra a aplicação do modelo em cenários práticos de moderação de conteúdos.

---

## Instalação e Requisitos

Para correr este projeto localmente, recomenda-se a criação de um ambiente virtual e a instalação das dependências.

Pré-requisitos de sistema:
* Python 3.8+
* NVIDIA GPU recomendada (mínimo 8GB VRAM) para inferência do modelo local.

Passos:
1. Clona o repositório:
git clone https://github.com/TEU_USER/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO

2. Instala as dependências:
pip install torch transformers datasets pandas scikit-learn bitsandbytes deep-translator

---

## Como Usar o Sistema de Inferência

Após o treino do modelo (ou tendo descarregado os pesos otimizados), podes testar a classificação de frases interativamente:

python testar_modelo.py

Exemplo de Utilização:

==================================================
Escreve uma frase para testar. Escreve 'sair' para fechar.
==================================================

Escreve uma frase: Estes migrantes vêm para o nosso país roubar os subsídios todos, deviam ser recambiados

Resultados análise
----------------------------------------
 -> D. O. Geral     : SIM (97.7%)
    Contra-discurso : Não (1.8%)
    D. Ofensivo     : Não (3.5%)
    Racializadas    : Não (6.5%)
    Ciganas         : Não (1.6%)
    LGBTQA+         : Não (0.4%)
 -> Migrantes       : SIM (92.9%)
----------------------------------------

---

## Contexto Académico
Este projeto foi desenvolvido como parte de uma Dissertação de Mestrado no Departamento de Informática da Universidade da Beira Interior (UBI), sob a orientação do Professor Doutor João Cordeiro. Todo o código e metodologia visam colmatar a escassez de recursos para o Processamento de Linguagem Natural (PLN) na variante europeia da língua portuguesa.
