import microphone
# import classifier

# classifier.load_graph("./models/CRNN/CRNN_L.pb")
# classifier.run_graph(wav_data=b"", labels="./models/labels.txt", num_top_predictions=3)
mic = microphone.Microphone()
microphone.list_devices()
