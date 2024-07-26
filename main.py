#!pip install transformers
import transformers
from transformers import GPT2Tokenizer,GPT2LMHeadModel

#!pip install datasets
#!pip install accelerate -U
#run one at each time and then restart runtime



model_name="gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

input_data=" python can be very"
max_length=100
num_return_sequences=1
input_ids = tokenizer(input_data,return_tensors="pt")["input_ids"]


output = model.generate(input_ids,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,  # Fixed parameter
                        temperature=0.7,
                        top_k=20,  # for more diverse sampling
                        no_repeat_ngram_size=2
                        )

#model.generate??

for i, sequence in enumerate(output):
    print(f"Generated sequence before fine tuning  {i + 1}: {tokenizer.decode(sequence)}")
    print("\nEnd of sequence \n")

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from transformers import TrainingArguments, TextDataset, DataCollatorForLanguageModeling, Trainer

from datasets import Dataset

#dataset = load_dataset('bookcorpus')

#dataset

#travel_keywords = r"(?i)\b(Python)\b"

#import re

#filtered_dataset = dataset["train"].filter(
 #   lambda x: re.compile(travel_keywords).search(x["text"]) is not None
#)


#filtered_dataset

#filtered_dataset.to_csv('travel.csv', index=False)

import pandas as pd

# Assuming your dataset is named "my_dataset.csv"
travel_data = pd.read_csv("/content/sample_data/python.csv")
travel_data =travel_data[:1000]

import re
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.lower()  # Convert to lowercase

travel_data['text'] = travel_data['text'].apply(clean_text)


# Clean travel descriptions
travel_data["cleaned_text"] = travel_data["text"].apply(clean_text)

# Create Hugging Face Dataset
train_dataset = Dataset.from_pandas(travel_data[["cleaned_text"]])



print(len(train_dataset))

train_dataset[10]

#with open("data.txt", "w") as f:
  #  for data in train_dataset["cleaned_text"]:
    #    f.write(" ".join(map(str, data)) + "\n")

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="/content/sample_data/python.csv",
    block_size=128,
)

#!pip install accelerate -U
#!pip install accelerate
#!pip install transformers[torch]

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust number of epochs
    per_device_train_batch_size=4,  # Adjust batch size based on GPU memory
    save_steps=200,
    save_total_limit=2,  # Save only the two best checkpoints
)

def load_data_collator(tokenizer, mlm = False):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
data_collator1=load_data_collator(tokenizer,False)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator1,
)

trainer.train()

trainer.save_model()
model2 = GPT2LMHeadModel.from_pretrained("/content/output")

model2 = GPT2LMHeadModel.from_pretrained("/content/output")

#model2??

output2 = model2.generate(input_ids,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,  # Fixed parameter
                        temperature=0.7,
                        top_k=20,  # for more diverse sampling
                        no_repeat_ngram_size=2
                        )

for i, sequence1 in enumerate(output2):
    print(f"Generated sequence after fine tuning {i + 1}: {tokenizer.decode(sequence1)}")
    print("\nEnd of sequence \n")

from datasets import load_metric


output2 = model2.generate(input_ids,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,  # Fixed parameter
                        temperature=0.7,
                        top_k=20,  # for more diverse sampling
                        no_repeat_ngram_size=2
                        )


prediction=[tokenizer.decode(sequence1)]
reference=["an image popped into his head glowing just like the grailshaped beacon in that monty python moviea bottle of bourbon "]

rouge=load_metric("rouge")

rouge.compute(predictions=prediction,references=reference)


!pip install rouge_score

from google.colab import drive
drive.mount('/content/drive')

