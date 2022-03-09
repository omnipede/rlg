from transformers import Wav2Vec2Tokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector, Wav2Vec2Model
import torch
import librosa
from datasets import load_dataset


def wav2vec2_sample():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # audio file is decoded on the fly
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    print(transcription)

    pass


def cc_generation():
    # Generate CC
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    input_audio, _ = librosa.load("/Users/omnipede/data/2Pac-Changes(OfficialMusicVideo)ft.Talent/clips/2Pac-Changes(OfficialMusicVideo)ft.Talent-99766-2235.mp4")

    input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print(transcription)

    pass


def embedding_extraction():

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

    # audio file is decoded on the fly
    inputs = feature_extractor(
        [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.embeddings


def embedding_extraction_2():

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # audio file is decoded on the fly
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state)

    pass
