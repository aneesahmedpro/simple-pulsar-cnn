CNN_INPUT_LENGTH = 200

CLASS_ID_TO_LABEL = {
    0: 'not pulsar',
    1: 'pulsar',
}

CLASS_LABEL_TO_ID = {v: k for k, v in CLASS_ID_TO_LABEL.items()}

BATCH_SIZE = 20
