# README: Classificação de dúvidas com DistilBERT

Este README fornece um guia para a utilização do DistilBERT na classificação de perguntas utilizando TensorFlow e a biblioteca Transformers. O código apresentado demonstra como carregar um modelo DistilBERT pré-treinado, ajustá-lo em um conjunto de dados personalizado para classificação de sequências e utilizá-lo de forma ajustada para fazer previsões.

## Pré-requisitos

Antes de executar o código, certifique-se de ter as dependências necessárias instaladas:

```bash
pip install transformers tensorflow pandas nltk scikit-learn
```

## Início
1. Preparação do Conjunto de Dados:
- Verifique a existência de um arquivo CSV (data.csv neste caso).
2. Pré-processamento dos Dados:
- Carregue e pré-processe o conjunto de dados usando o código fornecido.
- Tokenize o texto usando o tokenizador do DistilBERT.
3. Treinamento:
- Defina os argumentos de treinamento, como épocas, tamanhos de lote e diretórios de saída.
- Ajuste o modelo DistilBERT no conjunto de dados preparado.
```bash
model.fit(train_dataset.shuffle(1000).batch(16), epochs=7, validation_data=val_dataset.batch(64))
```
4. Avaliação:
- Avalie o modelo no conjunto de dados de validação.
```bash
evaluation_result = model.evaluate(val_dataset.batch(64))
```
5. Salvar Modelo e Tokenizador:
- Salve o modelo e o tokenizador ajustados para uso futuro.
```bash
save_directory = "/saved_models" 
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```
## Inferência
Agora que você possui um modelo e tokenizador ajustados, pode usá-los para fazer previsões em novos dados de texto. No caso em questão, foi inserido em um arquivo txt algumas perguntas de teste.

```bash
with open('perguntas.txt', 'r', encoding='utf-8') as file:
    perguntas = file.readlines()

for pergunta in perguntas:
    pergunta = pergunta.strip()  
    print(f"Pergunta: {pergunta}")
    predict_input = tokenizer_fine_tuned.encode(
        pergunta,
        truncation = True,
        padding = True,
        return_tensors = 'tf'    
    )

    output = model_fine_tuned(predict_input)[0]

    prediction_value = tf.argmax(output, axis = 1).numpy()[0]
    label_mapping = dict(zip(df['encoded_text'].astype('category'), df['label']))
    predicted_label = label_mapping[prediction_value]
    print(predicted_label)
```
