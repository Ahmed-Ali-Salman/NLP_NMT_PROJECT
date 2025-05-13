from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import os

#limit gpu mem
gpu = tf.config.list_physical_devices('GPU')
if gpu:
  try:
    tf.config.set_logical_device_configuration(
        gpu[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpu), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

app = Flask(__name__)
CORS(app)

# Models and tokenizers will be loaded here
eng_tokenizer = None
eng_tokenizer_ara = None
eng_tokenizer_fre = None
eng_tokenizer_ita = None
ara_tokenizer = None
fre_tokenizer = None
deu_tokenizer = None
ita_tokenizer = None
modela = None  # Arabic model
modelf = None  # French model
modeld = None  # German model
modeli = None  # Italian model
eng_length = 0
areng_length = 0
freng_length = 0
iteng_length = 0

# Helper function to get word from tokenizer
def get_word(index, tokenizer):
    # Handle zero index (padding)
    if index == 0:
        return None
        
    # Check if there's a reverse word index available (more efficient)
    if hasattr(tokenizer, 'index_word') and index in tokenizer.index_word:
        return tokenizer.index_word[index]
    
    # Fallback to the slower method
    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    
    return None

def load_models():
    global eng_tokenizer, eng_tokenizer_ara, eng_tokenizer_fre, eng_tokenizer_ita
    global ara_tokenizer, fre_tokenizer, deu_tokenizer, ita_tokenizer
    global modela, modelf, modeld, modeli
    global eng_length, areng_length, freng_length, iteng_length
    
    try:
        # Load models
        modeld = tf.keras.models.load_model('model/translation_modeld_v8.h5')
        modela = tf.keras.models.load_model('model/translation_modela_v6.h5')
        modelf = tf.keras.models.load_model('model/translation_modelf_v4.h5')
        modeli = tf.keras.models.load_model('model/translation_modeli.h5')
        
        # Load tokenizers
        import pickle
        with open('tokenizer/eng_tokenizer.pickle', 'rb') as handle:
            eng_tokenizer = pickle.load(handle)
        with open('tokenizer/eng_tokenizer_ara.pickle', 'rb') as handle:
            eng_tokenizer_ara = pickle.load(handle)
        with open('tokenizer/eng_tokenizer_fre.pickle', 'rb') as handle:
            eng_tokenizer_fre = pickle.load(handle)
        with open('tokenizer/eng_tokenizer_ita.pickle', 'rb') as handle:
            eng_tokenizer_ita = pickle.load(handle)
        with open('tokenizer/ara_tokenizer.pickle', 'rb') as handle:
            ara_tokenizer = pickle.load(handle)
        with open('tokenizer/fre_tokenizer.pickle', 'rb') as handle:
            fre_tokenizer = pickle.load(handle)
        with open('tokenizer/deu_tokenizer.pickle', 'rb') as handle:
            deu_tokenizer = pickle.load(handle)
        with open('tokenizer/ita_tokenizer.pickle', 'rb') as handle:
            ita_tokenizer = pickle.load(handle)
        
        # Set sequence lengths
        eng_length = 50
        areng_length = 50
        freng_length = 50
        iteng_length = 50
        
        print("Models and tokenizers loaded successfully!")
    except Exception as e:
        print(f"Error loading models or tokenizers: {str(e)}")
        raise

def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def translate_sentence(sentence, target_language):
    # Reference global variables
    global eng_tokenizer, eng_tokenizer_ara, eng_tokenizer_fre, eng_tokenizer_ita
    global ara_tokenizer, fre_tokenizer, deu_tokenizer, ita_tokenizer
    global modela, modelf, modeld, modeli
    
    if target_language == 'arabic':
        tokenizer = eng_tokenizer_ara
        modell = modela
        max_length = areng_length
        target_tokenizer = ara_tokenizer
    elif target_language == 'french':
        tokenizer = eng_tokenizer_fre
        modell = modelf
        max_length = freng_length
        target_tokenizer = fre_tokenizer
    elif target_language == 'german':
        tokenizer = eng_tokenizer
        modell = modeld
        max_length = eng_length
        target_tokenizer = deu_tokenizer
    elif target_language == 'italian':
        tokenizer = eng_tokenizer_ita
        modell = modeli
        max_length = iteng_length
        target_tokenizer = ita_tokenizer
    else:
        return "Target language not supported."
    
    # Validate required components are available
    if tokenizer is None:
        raise ValueError(f"Source tokenizer for {target_language} is not loaded")
    if modell is None:
        raise ValueError(f"Model for {target_language} is not loaded")
    if target_tokenizer is None:
        raise ValueError(f"Target tokenizer for {target_language} is not loaded")
    
    # Tokenize input sentence
    sentence = sentence.lower()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = sentence.strip()
    
    try:
        # Add debugging
        print(f"Processing with tokenizer: {type(tokenizer)}")
        tokens = tokenizer.texts_to_sequences([sentence])
        print(f"Tokens created: {tokens}")
        
        # Pad tokenized sequence
        padded_sequence = pad_sequences(tokens, maxlen=max_length, padding='post')
        print(f"Padded sequence shape: {padded_sequence.shape}")
        
        # Predict the translated sequence
        prediction = modell.predict(padded_sequence, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        
        # Decode the predicted sequence into text
        decoded_sentence = ''
        for token in prediction[0]:
            sampled_token_index = np.argmax(token)
            sampled_word = get_word(sampled_token_index, target_tokenizer)
            if sampled_word is None:
                break
            decoded_sentence += sampled_word + ' '
        
        return decoded_sentence.strip()
    except Exception as e:
        import traceback
        print(f"Error in translation process: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.get_json()
    if not data or 'text' not in data or 'target_language' not in data:
        return jsonify({'error': 'Missing text or target language'}), 400
    
    text = data['text']
    target_language = data['target_language'].lower()
    
    if not text:
        return jsonify({'translation': ''}), 200
    
    if 'modela' not in globals() or modela is None:
        mock_translations = {
            'arabic': 'مرحبا بالعالم',
            'french': 'Bonjour le monde',
            'german': 'Hallo Welt',
            'italian': 'Ciao mondo'
        }
        if target_language in mock_translations:
            return jsonify({'translation': mock_translations[target_language]}), 200
        else:
            return jsonify({'error': 'Unsupported language'}), 400
    
    try:
        # Debug info
        print(f"Translating: '{text}' to {target_language}")
        
        translated_text = translate_sentence(text, target_language)
        
        # Debug info
        print(f"Translation result: '{translated_text}'")
        
        return jsonify({'translation': translated_text}), 200
    except Exception as e:
        import traceback
        print(f"Translation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/languages', methods=['GET'])
def get_languages():
    languages = [
        {'code': 'arabic', 'name': 'Arabic'},
        {'code': 'french', 'name': 'French'},
        {'code': 'german', 'name': 'German'},
        {'code': 'italian', 'name': 'Italian'}
    ]
    return jsonify(languages), 200

if __name__ == '__main__':
    # Load models on startup
    try:
        load_models()
        print("Models loaded successfully! Server is ready.")
    except Exception as e:
        print(f"Warning: Could not load models: {str(e)}")
        print("Server will run in mock mode.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
