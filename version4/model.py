import time
import vertexai
from langchain_ollama import OllamaLLM
from tenacity import retry, stop_after_attempt, wait_random_exponential
from vertexai.preview.generative_models import GenerationConfig, GenerativeModel, HarmCategory, HarmBlockThreshold


PROJECT_ID = "GCP_PROJECT_ID"
LOCATION = "GCP_LOCATION"


class Model:
    def __init__(self, model_name):
        #'google' or 'vertex' or 'llama'.
        self.model_name = model_name
        
        # Gemini models
        if 'gemini' in model_name:
            self.provider = "google"
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            self.client = GenerativeModel(model_name)
        else:
            self.provider = 'ollama'
            self.llm = OllamaLLM(model=model_name)

    def invoke(self, prompt, **kwargs):
        if not prompt:
            return 'Contents must not be empty.'
        if self.provider == 'ollama':
            return self.llm.invoke(prompt, **kwargs)  
        elif self.provider == "google":
            return self.query_gemini(prompt, **kwargs)
        else:
            raise ValueError(f'Unsupported provider: {self.provider}')

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def query_gemini_with_retry(self, prompt, generation_config):
        safety_config = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        response = self.client.generate_content(prompt, generation_config=generation_config, safety_settings=safety_config)
        try:
            response_text = response.text
        except Exception as e:
            response_text = str(e)
        return response_text

    def query_gemini(self, prompt, rate_limit_per_minute = None, **kwargs):
        generation_config = GenerationConfig(
            stop_sequences=kwargs.get('stop', []),
            temperature=kwargs.get('temperature'),
            top_p=kwargs.get('top_p'),
        )
        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return self.query_gemini_with_retry(prompt, generation_config=generation_config)

    




if __name__ == '__main__':
    def test_model(model_name, prompt):
        print(f'Testing model: {model_name}')
        model = Model(model_name)
        print(f'Prompt: {prompt}')
        response = model.query(prompt)
        print(f'Response: {response}')
        num_tokens = model.get_token_count(prompt)
        print(f'Number of tokens: {num_tokens}')
    prompt = 'Hello, how are you?'
    for model in ['gpt-4o-mini-2024-07-18', 'gemini-1.5-flash']:
    # for model in ['mistralai/Mistral-Nemo-Instruct-2407']:
        test_model(model, prompt)