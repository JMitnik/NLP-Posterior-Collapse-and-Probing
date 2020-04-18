import torch

def generate_next_words(
    model,
    custom_data,
    start_sent="as they parked out front and owen stepped out of the car , he could see",
    device='cpu',
    max_length=10,
    temperature=1
):
    print(f'Start of the sentence: {start_sent} || Max Length {max_length} .')
    with torch.no_grad():
        encoded_start = custom_data.tokenizer.encode(start_sent, add_special_tokens=True)[:-1]
        sentence = encoded_start
        print(sentence)

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

        print(sentence)
        print(f'Sentence Length: {len(sentence)}')

        return sentence


def evaluate_model(
    model,
    data_loader,
    epoch,
    device,
    criterion
):
    model.eval()
    total_loss: float = 0

    for batch in data_loader:
        with torch.no_grad():
            input = batch[:, 0: -1].to(device)
            target = batch[:, 1:].to(device)

            output = model(input)

            loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))
            total_loss += loss / len(batch)

    total_loss = total_loss / len(data_loader)
    return total_loss, torch.log(torch.tensor(total_loss))
