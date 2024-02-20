from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopw = stopwords.words('english')

caminho = 'data.csv'
df = pd.read_csv('data.csv', sep=';', encoding='latin1', names=["label", "question"])

df['count'] = df['question'].apply(lambda x: len(x.split()))

category_count = df['label'].value_counts()

categories = category_count.index


df['encoded_text'] = df['label'].astype('category').cat.codes
#print(df)
data_question = df['question'].to_list()

data_labels = df['encoded_text'].to_list()

train_question, val_question, train_labels, val_labels = train_test_split(data_question, data_labels, test_size = 0.2, random_state = 0 )

train_question, test_question, train_labels, test_labels = train_test_split(train_question, train_labels, test_size = 0.1, random_state = 0 )

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_question, truncation = True, padding = True)

val_encodings = tokenizer(val_question, truncation = True, padding = True )

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))


val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=16)

from transformers import TFDistilBertForSequenceClassification, TFTrainingArguments


training_args = TFTrainingArguments(
    output_dir='./results',          
    num_train_epochs=7,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=1e-5,               
    logging_dir='./logs',            
    eval_steps=100                   
)
with training_args.strategy.scope():
    trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 16)



learning_rate = training_args.learning_rate


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=16)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


model.fit(train_dataset.shuffle(1000).batch(16), epochs=7, validation_data=val_dataset.batch(64))


evaluation_result = model.evaluate(val_dataset.batch(64))

save_directory = "/saved_models" 

model.save_pretrained(save_directory)

tokenizer.save_pretrained(save_directory)

tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)

model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)


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
    #print(prediction_value)
    print(predicted_label)
