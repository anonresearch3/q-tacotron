class Hparams():
    """Create model hyperparameters. Parse nondefault from given string."""
    ################################
    # Experiment Parameters        #
    ################################
    epochs=2
    iters_per_checkpoint=1000
    seed=1234

    cudnn_enabled=True
    cudnn_benchmark=False
    ignore_layers=['embedding.weight']
    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk=False
    training_files='filelists/ljs_audio_text_train_filelist.txt'
    validation_files='filelists/ljs_audio_text_val_filelist.txt'
    text_cleaners=['english_cleaners']
    path_to_dataset='./dataset/LJSpeech-1.1/'
    data_for_train="data_quantized.json"
    path_to_tmp = 'tmp'
    path_to_letter_durations = 'aggregated_durations_lj.json'
    ################################
    # Audio Parameters             #
    ################################
    max_wav_value=32768.0
    sampling_rate=22050
    filter_length=1024
    hop_length=256
    win_length=1024
    n_mel_channels=80
    mel_fmin=0.0
    mel_fmax=8000.0
    min_text_len = 1
    min_spec_len = 1
    max_text_len = 400
    max_spec_len = 2000
    ################################
    # Model Parameters             #
    ################################
    symbols = list('_=-~!\'"(),.:;? abcdefghijklmnopqrstuvwxyz1234')
    ru_symbols = list('_ =~*.,;:?!-()\"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАЯОЁУЮЫИЭЕ^&$1234')
    n_symbols=len(symbols)
    symbols_embedding_dim=512

    # Encoder parameters
    encoder_kernel_size=5
    encoder_n_convolutions=3
    encoder_embedding_dim=512

    # Decoder parameters
    n_frames_per_step=1  # currently only 1 is supported
    decoder_rnn_dim=1024
    prenet_dim=256
    max_decoder_steps=2000
    gate_threshold=0.5
    p_attention_dropout=0.1
    p_decoder_dropout=0.1

    # Attention parameters
    attention_rnn_dim=1024
    attention_dim=128

    # Location Layer parameters
    attention_location_n_filters=32
    attention_location_kernel_size=31

    # Mel-post processing network parameters
    postnet_embedding_dim=512
    postnet_kernel_size=5
    postnet_n_convolutions=5

    # TPGST
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    E = 512
    token_num = 10
    num_heads = 8
    emb = 312
    text_encoder_embedding_dim = 384

    embedding_shapes = [8, 16, 8, 16, 16]
    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate=False
    learning_rate=1e-3
    weight_decay=1e-6
    grad_clip_thresh=1.0
    batch_size=96
    mask_padding=True  # set model's padded outputs to padded values


