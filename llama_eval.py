from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent, AgentType

def correct_grammar_and_spelling_with_llm(text):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=""" 
        You are an expert proofreader. Please correct the grammar and spelling mistakes in the following text:

        {text}

        Provide the corrected text below:
        """
    )

    llm = OllamaLLM(model="llama3.2:3b")
    chain = prompt | llm
    corrected_text = chain.invoke({"text": text})
    print("Corrected text: ",corrected_text)
    return corrected_text

grammar_and_spelling_correction_tool = Tool(
    name="Grammar and Spelling Correction",
    func=correct_grammar_and_spelling_with_llm,
    description="Correct the grammar and spelling of a given text using an LLM"
)
def evaluate_answer_with_llm(question, answer):
    llm = OllamaLLM(model="llama3.2:3b")
    prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    You are an expert interviewer tasked with evaluating a candidate's answer. 
    Follow these steps carefully to ensure a clear and comprehensive evaluation:

    Step 1: **Summarize the Candidate's Answer** 
    - Provide a concise summary of the candidate's response. 
    - Highlight the key points and concepts they addressed.

    Step 2: **Evaluate the Response** 
    - Assess how well the answer addresses the question. Consider factors such as accuracy, relevance, clarity, and depth.
    - Comment on strengths (what was done well) and areas for improvement (what could be better). 
    - Take into account the context of the interview, such as the need for brevity and clarity.

    Step 3: **Provide a Score and Justification** 
    - Assign a score out of 10 based on these criteria:
      - Clarity and relevance of the explanation
      - Depth and accuracy of knowledge demonstrated
      - Ability to organize and articulate the response within a time-limited setting
    - Explain the reasoning behind the score, including specific aspects that influenced your evaluation.

    Question: {question}
    Answer: {answer}

    Please respond using the following format:
    1. **Summary of the Answer:** [Your summary]
    2. **Evaluation:** [Your evaluation]
    3. **Score:** [Your score out of 10 and rationale for the score]
    """
)


    chain = prompt | llm
    result = chain.invoke({"question": question, "answer": answer})
    return result



def agent_to_process_answer(question, answer):
    tools = [grammar_and_spelling_correction_tool]
    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=OllamaLLM(model="llama3.2:3b"),
        verbose=True
    )
    
    corrected_answer = agent.invoke(f"Correct the grammar and spelling of the answer: {answer}")
    evaluation_result = evaluate_answer_with_llm(question, corrected_answer)
    
    return evaluation_result



# if __name__ == "__main__":
#     question = "What are your strengths?"
#     answer = "I am hardworking and good problem solver."

#     evaluation_result = agent_to_process_answer(question, answer)
#     print("Evaluation Result: ")
#     print(evaluation_result)
