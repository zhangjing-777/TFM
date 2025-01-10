import pandas as pd

def extract_code_and_execute(answer, df):
           
    if "```python" not in answer:
        print("Error: The answer does not contain a valid Python code block.") 
        return None 
        
    # Extract code block
    code_start = answer.find("```python") + 9
    code_end = answer.find("```", code_start)
    code = answer[code_start:code_end].strip()
    
    try:
        # Create local namespace and execute code
        local_dict = {'df': df, 'pd': pd}
        code = f"result = {code}"
        exec(code, None, local_dict)
        return local_dict.get('result')
                
    except SyntaxError as syntax_error:
        print(f"Code execution error: Invalid syntax in code: {code}\nError: {str(syntax_error)}")
        return None 
    
    except Exception as code_error:
        print(f"Code execution error: {str(code_error)}")
        return None 
            
def format_llm_output(output: str) -> str:

    sections = output.split('\n\n')   
    formatted_output = []
    headers = [
        '**The question:**',
        '**The relative result:**',
        '**The concluding response:**'
    ]

    for section in sections:
        for header in headers:
            if header in section:
                formatted_output.append(section.replace(header, f'\n{header}').strip())
                break
        else:
            formatted_output.append(section.strip())
    
    return '\n\n'.join(formatted_output)