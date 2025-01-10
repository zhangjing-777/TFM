from execute import extract_code_and_execute, format_llm_output


class RAGAgent:
    def __init__(self, retriever, prompt, model, df):
        self.retriever = retriever
        self.prompt = prompt
        self.model = model
        self.df = df

    def processor(self, query, **kwargs):
        context = self.retriever.retrieve_schema(query, self.df)
        prompt_output = self.prompt.format(context=context, question=query)
        model_output = self.model.invoke(prompt_output, **kwargs)
        return model_output
    
    def invoke(self, query, **kwargs):
        result = None
        max_attempts = 3  
        attempts = 0
        
        while result is None and attempts < max_attempts:
            code = self.processor(query, **kwargs)
            result = extract_code_and_execute(code, self.df)
            attempts += 1  
        
        return {'code': code, 'result': result}

    
class InterpAgent:
    def __init__(self, prompt, model):
        self.prompt = prompt
        self.model = model

    def invoke(self, context, query, **kwargs):
        prompt_output = self.prompt.format(context=context, question=query)
        model_output = self.model.invoke(prompt_output, **kwargs)
        return format_llm_output(model_output)
  