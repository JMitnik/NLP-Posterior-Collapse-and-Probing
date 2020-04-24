import torch
import torch.distributions as D
import torch.nn.functional as F
from models import VAE
import numpy as np

def generate_next_words(
    model,
    custom_data,
    start_sent="as they parked out front",
    device='cpu',
    max_length=10,
    temperature=1
):
    model.eval()

    print(f'Start of the sentence: {start_sent} || Max Length {max_length} .')

    with torch.no_grad():
        encoded_start = custom_data.tokenizer.encode(start_sent, add_special_tokens=True)[:-1]
        sentence = encoded_start

        for i in range(max_length):
            # Create input for the model
            model_inp = torch.tensor(sentence).to(device)
            model_inp = model_inp.unsqueeze(0) # Ensures we pass a 1(=batch-dimension) x sen-length vector

            output = model(model_inp).cpu().detach()

            # print(output)
            prediction_vector = F.softmax(output[0][-1] / temperature)
            sample_vector = D.Categorical(prediction_vector)
            sample = int(sample_vector.sample())
            if sample == 3: # cannot produces UNK token
                i = i-1
                continue
            sentence.append(sample)

            if sample == 2: # If we sampled EOS
                break

        return sentence
