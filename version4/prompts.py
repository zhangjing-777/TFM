from langchain.prompts import PromptTemplate


def get_prompt(template):                
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]  
    )
    return prompt
  
  
template1 = """You are a pandas dataframe query code generator.The name of the dataframe is `df`. Your task is to answer the question: {question} with pandas dataframe operation code.

              Provide concise answers with only 1 section:
                Code: [write one line of pandas dataframe operation code in ```python``` block, 
                          Example - code:
                                    ```python
                                        df['category_name'].value_counts().count()
                                    ```
                          ]
              Rules:
                1. Understand the meaning of the user's question correctly and choose proper column names from {context} to generate pandas dataframe operation code.
                2. If the question refers to a summary or aggregate of multiple columns (like total/comprehensive scores), infer the appropriate calculation based on the context.
                3. Maybe the keywords of the user's question are not the same as the column names of `df`, please use similar semantic matching.
                4. Ensure that the generated code is syntactically correct and complete, including all necessary parentheses and syntax.
                5. The purpose of generating code is to produce data that can answer the user's question. If it is not possible to give a code for direct data based on the question, then giving a code that can obtain detailed information will suffice.
                6. Just return complete executable code in ```python``` block using the existing `df`.
                7. No explanations or additional text.
                
              You can use these pandas operations:
                1. Basic statistics: df.describe(), df[column].mean(), df[column].max(), etc.
                2. Group statistics: df.groupby(column).agg()
                3. Sorting: df.sort_values(by=column)
                4. Filtering: df[df[column] > value]
                5. Aggregating multiple columns: df[['col1', 'col2', 'col3']].sum(axis=1) for total scores, etc.
                ...
                
              Example:
                question: "If food impacts writing score?"
                Answer: "```python\ndf.groupby('lunch')['writing score'].mean()\n```"
                """
                
template2 = """You are a pandas dataframe query code generator. The name of the dataframe is `df`. Your task is to answer the question with pandas dataframe operation code.
               
               The question is: {question}
               The context is: {context}
               
              **Important Note:**
              You only have a context that includes the column names and dtypes of the columns. There is no row data or cell data information available. Therefore, when the user's question involves specific cell information, you should provide the code that corresponds to the entire column's data instead of specific values.
              
              **Example:**
              If the question is: "If students who completed preparation have a better writing score?", 
              the answer should be: 
              ```python
              df.groupby('test preparation course')['writing score'].mean()
              ```
              This is because you do not know the specific values of 'test preparation course', and using them could lead to errors.

              Provide concise answers with only 1 section:
                Code: [write one line of pandas dataframe operation code in ```python``` block, 
                          Example - code:
                                    ```python
                                        df['category_name'].value_counts().count()
                                    ```
                          ]
              Rules:
                1. Understand the meaning of the user's question correctly and choose proper column names to generate pandas dataframe operation code.
                2. Consider the column types when writing code.For example,you cannot use the `cor()` function between `object` type columns and `int` type columns.
                3. If the question refers to a summary or aggregate of multiple columns (like total/comprehensive scores), infer the appropriate calculation based on the context.
                4. If the keywords of the user's question are not the same as the column names of `df`, please use similar semantic matching.
                5. Ensure that the generated code is syntactically correct and complete, including all necessary parentheses and syntax.
                6. The purpose of generating code is to produce data that can answer the user's question. If it is not possible to give a code for direct data based on the question, then giving a code that can obtain detailed information will suffice.
                7. Just return complete executable code in ```python``` block using the existing `df`.
                8. No explanations or additional text.
                
              You can use these pandas operations:
                1. Basic statistics: df.describe(), df[column].mean(), df[column].max(), etc.
                2. Group statistics: df.groupby(column).agg()
                3. Sorting: df.sort_values(by=column)
                4. Filtering: df[df[column] > value]
                5. Aggregating multiple columns: df[['col1', 'col2', 'col3']].sum(axis=1) for total scores, etc.
                ...
                """
                
                
#指示式则更适合需要快速、清晰输出的场景。  
#few-shot & COT           
combined_template = """You are a pandas dataframe query code generator. The name of the dataframe is `df`. Your task is to answer the question with pandas dataframe operation code.

              The question is: {question}
              The context is: {context}
              
              **Thinking Process:**
              1. Identify the key information needed to answer the question.
              2. Determine which columns in the dataframe are relevant to the question.
              3. Consider the appropriate pandas operation code and the types of these columns to ensure the correct code is applied.
              4. Formulate the final code in ```python``` block.
               
              **Important Note & Examples:**
              Note1:You only have a context that includes the column names and dtypes of the columns. There is no row or cell information available. Therefore, when the user's question involves specific cell information, you should provide the code that corresponds to the entire column's data instead of specific values.
              Example1:
                 Question: "If students who completed preparation have a better writing score?"
                 Thinking:
                   1. I need to find the 'writing score' for completed preparation and the 'writing score' for not completed preparation separately.
                   2. I need to use 'writing score' column and 'preparation' column, but there is no 'preparation' column in the `df`, so I need to choose a similar column 'test preparation course' to alternative.
                   3. I haven't the values of 'test preparation course' column, so I will get all values with grouping by 'test preparation course' column and using mean() on 'writing score' column,and the type of 'writing score' is `int64`,the type of 'test preparation course' is `object`, that will be a valid operation.
                   4. My final code is: df.groupby('test preparation course')['writing score'].mean(), and I need to return the code in ```python``` block. 
                 Answer: 
                 ```python
                 df.groupby('test preparation course')['writing score'].mean()
                 ```
                 
              Note2:If the keywords of the user's question are not the same as the column names of `df`, please use similar semantic matching.
              Example2:
                 Question: "What is the average arithmetic score?"
                 Thinking:
                   1. I need to find the average value of the 'arithmetic score' column.
                   2. There is no 'arithmetic score' column in the `df`, but there is a 'math score' column that is similar, so I can use the 'math score' column to calculate the average value.
                   3. I can use the `mean()` function, and the type of 'math score' is `int64`, that will be a valid operation for the column.
                   4. My final code is: df['math score'].mean(), and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df['math score'].mean()
                 ```
                 
              Note3:If the question refers to a summary or aggregate of multiple columns (like total/comprehensive scores), infer the appropriate calculation based on the context.
              Example3:
                 Question: "What's the best comprehensive score?"
                 Thinking:
                   1. I need to find the maximum value of comprehensive score.
                   2. There isn't 'comprehensive score' column in the `df`, but 'comprehensive score' can be the sum of 'reading score', 'writing score', and 'math score'.
                   3. I can use the `max()` and `sum()`function, and 'reading score' 'writing score' 'math score' are all of type `int64`,that will be a valid operation.
                   4. My final code is: df[['reading score', 'writing score', 'math score']].sum(axis=1).max(), and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df[['reading score', 'writing score', 'math score']].sum(axis=1).max()
                 ```
                 
              Note4:The purpose of generating code is to produce data that can answer the user's question. If it is not possible to give a code for data based on the question directly, then giving a code that can obtain detailed information will suffice.
              Example4:
                 Question: "What're the features of the student who has the best reading score?"
                 Thinking:
                   1. I need to find the features of the student who has the maximum reading score.
                   2. I can use 'features' column and 'reading score' column, but there isn't 'features' column in the `df`, and `features` can be the combination of all columns.
                   2. I can use the `idxmax()` function and `iloc` function to get the row of the student who has the maximum reading score, and the type of 'reading score' is `int64`, that will be a valid operation.
                   3. My final code is: df.iloc[df['reading score'].idxmax()], and I need to return the code in ```python``` block.
                 Answer: 
                 ```python
                 df.iloc[df['reading score'].idxmax()]
                 ```
                 
              Rules:
                1. Consider the column types when writing code.For example,you cannot use the `cor()` function between `object` type columns and `int` type columns.
                2. Ensure that the generated code is syntactically correct and complete, including all necessary parentheses and syntax.
                3. Just return complete executable code in ```python``` block using the existing `df`, and don't include any explanations or additional text.
       
              Now, please follow this thinking process to generate the code for the following question:
              Question: {question}

              **Answer:**
              ```python
              [write your pandas operation code here]
              ```
              """

interp_template ="""
      You are an information analysis and summary assistant.
      Your task is to provide a concluding and logical response according to the question and the relative result.

      the question: {question}
      the relative result: {context}

      Please provide a clear and logical response to the user, including 3 sections and no others:
            - The question: {question}
            - The relative result: {context}
            - The concluding response: a concluding response that is reasonable and reliable based on the user's question and the relative result.

            example:
               The question: Which racial group has the best writing score?
               The relative result: ['code': "```python\ndf.loc[df['writing score'].idxmax()]['race/ethnicity']\n```",
                                    'result': 'group D']
               The concluding response: The relative result indicates that Group D has achieved the highest writing scores, making it the racial group with the best writing performance.
            
      """
      
      
"""系统就变为了，第一部分生成明细，减少搜索信息；第二部分再用大模型生成更自然的回答。优势为问题拆分，一步步解决
第二部时,可以根据result为None或值为无法理解的内容时要求重新执行第一步。

"""