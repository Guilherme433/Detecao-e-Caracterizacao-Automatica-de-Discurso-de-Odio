import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

caminho_treino = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_treino_retrotraducao.csv"
caminho_val = r"D:\Projetos\Projeto_final_CURSO\Projeto\dados\csv\dataset_validacao_oficial.csv"

modelo_nome = "PORTULAN/albertina-100m-portuguese-ptpt-encoder"

classes = [
    'D. O. Geral',
    'Contra-discurso', 'D. Ofensivo', 
    'Racializadas', 'Ciganas', 'LGBTQA+', 'Migrantes'
]

print("A carregar conjunto de Treino e Validação")

df_train = pd.read_csv(caminho_treino)
df_train = df_train.dropna(subset=['texto'] + classes)
train_texts = df_train['texto'].tolist()
train_labels = df_train[classes].astype(float).values.tolist()

df_val = pd.read_csv(caminho_val)
df_val = df_val.dropna(subset=['texto'] + classes)
val_texts = df_val['texto'].tolist()
val_labels = df_val[classes].astype(float).values.tolist()

print(f"-> Frases para Estudar (Treino): {len(train_texts)}")
print(f"-> Frases para Simulacro (Validação): {len(val_texts)}")

tokenizer = AutoTokenizer.from_pretrained(modelo_nome)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

class HateSpeechMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateSpeechMultiLabelDataset(train_encodings, train_labels)
val_dataset = HateSpeechMultiLabelDataset(val_encodings, val_labels)

def compute_metrics(pred):
    labels = pred.label_ids.astype(int) 
    
    probs = torch.sigmoid(torch.tensor(pred.predictions))
    preds = (probs > 0.5).int().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy_exata': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

if __name__ == '__main__':
    print(f"\nA inicializar o {modelo_nome}")
    
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_nome, 
        num_labels=len(classes),
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir='./resultados_modelo_v1',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none",      
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        fp16=False 
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n" + "="*50)
    print("A começar o treino do modelo.")
    print("="*50)

    trainer.train()

    caminho_guardar = r"D:\Projetos\Projeto_final_CURSO\Projeto\modelo_v1_7classes"
    trainer.save_model(caminho_guardar)
    tokenizer.save_pretrained(caminho_guardar)

    print(f"\nModelo guardado em:\n{caminho_guardar}")