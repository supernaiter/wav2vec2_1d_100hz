from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer, AutoModelForCTC, HubertForCTC, AutoProcessor
import random
from datasets import load_dataset
import torch

checkpoint_path = "results/wav2vec2-pretrained_10000epochs_32batch_2022-07-02_20-22-04/"
model=AutoModelForCTC.from_pretrained(checkpoint_path).cuda()

# とりあえずprocessorを持ってくる
processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo-google-colab")

# librispeech datasetでの評価
ls = load_dataset("librispeech_asr", "clean", split="validation")

num  = 0

try:
  while True:
    input()
    num = random.randint(0, len(ls)-1)    
    y = ls["audio"][num]["array"] 
    input_values = processor(y, sampling_rate=16_000, return_tensors="pt").input_values  # Batch size 1

    logits = model(torch.tensor(input_values, device="cuda")).logits

    predicted_ids = torch.argmax(logits.to("cuda"), dim=-1).clone().detach()    
    transcription = processor.decode(predicted_ids[0])
    print("transcription", transcription)
    print("labeled:", ls["text"][num])
  

except keyboardinterrupt:
  exit()
