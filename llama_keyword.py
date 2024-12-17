from langchain_ollama import OllamaLLM


# Initialize LLaMA
llm = OllamaLLM(model="llama3.2:3b")
# You are an expert AI assistant. Extract relevant keywords or phrases from the given query and categorize them into appropriate keys. 
#     Output the result **only** as a valid Python dictionary where keys represent categories like 'skills', 'location', 'experience', 'role', etc., 
#     and values are lists of relevant keywords.
def extract_and_categorize_keywords_with_llama(query):
    prompt = f"""
    
    You are an expert AI assistant. Extract the relevant keywords or phrases from the given query. 
    Provide them in a comma-separated format without additional explanations.
    Query: "{query}"

    Python Dictionary:
    """
    response = llm.invoke(prompt).strip()
    print(response)
    if response.startswith("{") and response.endswith("}"):
        try:
            return eval(response) 
        except SyntaxError:
            raise ValueError("not a valid Python dictionary.")
    else:
        raise ValueError("LLM did not return a dictionary.")


query = "AI developer in Noida 3 to 7 years of experience skills in ML, gen Ai, llm, langchain, data science"

try:
    categorized_keywords = extract_and_categorize_keywords_with_llama(query)
    print("Categorized Keywords:\n", categorized_keywords)
except ValueError as e:
    print("Error:", e)
