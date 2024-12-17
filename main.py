import time
import winsound
from text_to_speech import text_to_speech
from audio_stream_transcribe import transcribe_audio
from llama_eval import evaluate_answer_with_llm

def process_question(question):
    text_to_speech(question)
    time.sleep(5)
    for _ in range(2):
        winsound.Beep(1000, 1000)
        time.sleep(1)
    answer = transcribe_audio()
    print(f"\nAnswer: {answer}")
    evaluation = evaluate_answer_with_llm(question, answer)
    print("\nEvaluation Result:", evaluation)

def run_interview_system():
    print("AI Voice Interview System")
    print("Enter questions to start. Type 'exit' to quit.")
    while True:
        question = input("\nYour Question: ")
        if question.strip().lower() == 'exit':
            print("Exiting... Goodbye!")
            break
        process_question(question)

if __name__ == "__main__":
    run_interview_system()
