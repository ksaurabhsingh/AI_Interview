from flask import Flask, request, jsonify
from audio_stream_transcribe import transcribe_audio
from llama_eval import evaluate_answer_with_llm
from text_to_speech import text_to_speech
import winsound 
import time
app = Flask(__name__)
@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400

        text_to_speech(question)
        winsound.Beep(1000, 1000)  
        time.sleep(1)
        winsound.Beep(1000, 1000) 
        answer = transcribe_audio()
        evaluation = evaluate_answer_with_llm(question, answer)

        return jsonify({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
