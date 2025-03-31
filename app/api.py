from flask import Flask, request, jsonify
from sbc.endecode import sbc_encoder as SbcEncoder, sbc_decoder as SbcDecoder
import os
import tempfile

#Define Flask
App = Flask(__name__)

#Endpoint [POST] /encode
@App.route('/encode', methods=['POST'])
def Encode():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    quality = request.form.get('q', 4, type=int)
    transferType = request.form.get('transfer', '709')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tempInput:
        inputPath = tempInput.name
        file.save(inputPath)
    outputPath = tempfile.mktemp(suffix=".sbc")
    try:
        SbcEncoder(inputPath, quality, outputPath, transferType)
        return jsonify({'message': 'Encoding completed', 'outputPath': outputPath})
    except Exception as error:
        return jsonify({'error': str(error)}), 500
    finally:
        os.remove(inputPath)

#Endpoint [POST] /decode
@App.route('/decode', methods=['POST'])
def Decode():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    playOption = request.form.get('play', 0, type=int)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sbc") as tempInput:
        inputPath = tempInput.name
        file.save(inputPath)
    try:
        SbcDecoder(inputPath, playOption)
        return jsonify({'message': 'Decoding completed'})
    except Exception as error:
        return jsonify({'error': str(error)}), 500
    finally:
        os.remove(inputPath)

#Run Flask
if __name__ == '__main__':
    App.run(host='0.0.0.0', port=8080, debug=True)