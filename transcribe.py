import streamlit as st
#import plotly.express as px
#import networkx as nx
#import plotly.graph_objects as go
import whisperx
import gc
import torch
import librosa




def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    device = "cpu"
    #audio_file = "/content/230802_0003.wav" #CHANGE THIS TO YOUR FILE NAME (after uploading file)
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
    audio_file= st.file_uploader("Choose a file")

    # 1. Transcribe with original whisper (batched)
    if "model" not in st.session_state:   
        st.session_state["model"] = whisperx.load_model("small", device, compute_type=compute_type, language = 'en')
    if audio_file != None:
        model = st.session_state["model"]

        audio,sr = librosa.load(audio_file)
        #audio = whisperx.load_audio(audio_file.read())
        result = model.transcribe(audio, batch_size=batch_size, language = 'en')
        #print(result["segments"]) # before alignment

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        print(result["segments"]) # after alignment

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=st.secrets["hf"], device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model({"waveform": audio, "sample_rate": sr},min_speakers=2, max_speakers=2)
        # diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        for i in result['segments']:
            st.write(f"{i['speaker']}: {i['text']}")