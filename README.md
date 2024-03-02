# Finetune_LLM_Gemma-2b-it
![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/bf676a2d-7d9a-4618-9b89-89c044546654)

## Description
- Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.
- In this project, I want to finetune the model in order to generate code.

## Fine-tuning Process
- I finetune Gemma-2b-it by T4 (Google Colab) which have 15GB GPU Ram. However, according to my observations, this task only need around 10GB.
- I use low-rank adapters (LoRA) using get_peft_model utility function and the prepare_model_for_kbit_training method from PEFT.
- I use the dataset [TokenBender/code_instructions_122k_alpaca_style](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style).
Preview:

![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/c30ccd67-3073-489c-ab10-b9cafffa6f2e)

## Why finetune?
- I request the model to code the fibonacci series in python using reccursion. And here is result before fine-tuning:
~~~
user
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  code the fibonacci series in python using reccursion
  
model
  

  def fibonacci(n):
    """
    Calculates the nth number in the Fibonacci sequence.

    Args:
      n: The index of the Fibonacci number to calculate.

    Returns:
      The nth Fibonacci number.
    """

    # Base case: If n is 0 or 1, return the corresponding Fibonacci number.
    if n == 0:
        return 0
    elif n == 1:
        return 1

    # Calculate the previous two Fibonacci numbers.
    previous = fibonacci(n-1)
    next = fibonacci(n-2)

    # Return the Fibonacci number.
    return previous + next
```
~~~
- The answer is quite good but I expected something like this (from GeeksforGeeks): ![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/f4f790e7-e479-4c7a-bbcb-550cd93c4baf)

## Setting Model Parameters and Lora 
- I set various parameters for our fine-tuning process, including QLoRA (Quantization LoRA) parameters, bitsandbytes parameters, and training arguments.
- [Here](https://twitter.com/adithya_s_k/status/1744065797268656579?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1744065797268656579%7Ctwgr%5E367f3db066e6dfcc3b91caa1d2133fa0d118e285%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fcdn.embedly.com%2Fwidgets%2Fmedia.html%3Ftype%3Dtext2Fhtmlkey%3Da19fcc184b9711e1b4764040d3dc5c07schema%3Dtwitterurl%3Dhttps3A%2F%2Ftwitter.com%2Fadithya_s_k%2Fstatus%2F1744065797268656579image%3D) is a tweet on how to pick the best Lora config:


![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/8eef6d9f-0084-4845-b42e-b8f94dff6940)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">If you have ever fine-tuned LLMs using Lora and felt disappointed after merging the adapter with the base model, heres a reason why that might have happened:<br><br>👉🏼 Improper r and alpha values. These are the most important values that are determined by the purpose behind your…</p>&mdash; Adithya S K (@adithya_s_k) <a href="https://twitter.com/adithya_s_k/status/1744065797268656579?ref_src=twsrc%5Etfw">January 7, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

- This is my config:
  
![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/183c476d-ee6a-4724-badb-06f1cf43f360)

- Number of trainable parameters:

![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/a88fc38c-0484-4c3f-8951-9ca91d7f8844)



## Run the training
- I ran for 100 steps.
- Loss:
  
![image](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/a8be0990-92a0-4258-aa8a-2abfa25e0b3d)
 
## Result
- Here is result of model after fine-tuning:

~~~  user
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  code the fibonacci series in python using reccursion
  
model
  

   
modeldef fibonacci(first, second):
    if first > second:
        return fibonacci(second, first)

    return first + second

if __name__ == '__main__':
    result = fibonacci(0, 1)
    print(result) 
modeldef createFibSequence(n):
    fibList = []
    # First two number are initialized 
    fibList.append(0)
    fibList.append(1) 

    # Calculate and add to the list 
    for i in range(1, n):
        fibList.append(fibList[i - 1] + fibList[i - 2])
    return fibList
  
fibList = createFibSequence(8)
print(fibList) 
modelHere are two ways to calculate the Fibonacci series in Python using recursion: 

```python
def fibonacci(first, second, n):
    if first > second:
        return fibonacci(second, first)
    return first + second

# Access the sequence and print the first 5 terms 
for i in range(1, 20):
        print(fibonacci(-5, 0, i)) 
```

Here is the code to create an algorithm to get the fibonacci series for a given number n: 
```python
def fibonacci(n):
    return [0 for i in range(n+1)] 
  
    first = 0
    second = 1 
    for i in range(1, n+1):
        a = second
        b = first + second
        first = second
        second = a + b 
    return second 
``` 
"""
This code implements a function to generate the sequence of numbers, 
that is the Fibonacci sequence. 
"""

first = [0,1]
while len(first) < n: 
    first.append(first[0] + first[1]) 
    second = [0,1]
    while len(second) < n: 
        a = second[0] + second[1]
        second.append(a)
        second.append([first[0],first[0]])
    n = len(first)
 
print(first[n])  
```  
Here is another way to calculate the Fibonacci series, without using recursion. 
```python 
def fibonacci(n):
    if n <= 1:
        return n
    oldList = [0,1] 
    newList = [0,1] 
    while len(oldList) < n:
        newNumber = oldList[0] + oldList[1]
        newList.append(newNumber)   
        oldList.append(newNumber) 
        n -= 1 
    return newList[-1] 
```         
## Explanation of the code 
1. The first loop in the code generates the first `n` numbers in the Fibonacci sequence. 
2. The second loop updates the `first` and `second` lists to generate the next sequence. 
3. Each list generated by the second loop is appended to the `last` list. 
4. The `last` list when it is filled, produces the first `n` numbers in the Fibonacci sequence. 

## Time complexity o the code
The code above will need O(n) time to produce the first n numbers in the Fibonacci sequence, because it iterates over the whole sequence. 

## Space complexity o(n)
The code also needs O(n) space to store the two lists used for generating the sequence, plus the list used to store the results. o(n) space complexity is not a practical way of producing the sequence of numbers.             
 
This code is not practical if n is not 0, because it will be very slow to reach the first numbers. 

## Improved version 
Here is an improved version of the code, which can be used if n is either 0 or 1: 
```python
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n-1) + fibonacci(n-2) 
``` 
Here are
~~~
- The result (in improved version) can be acceptable, it looks quite similar to the solution of GeeksforGeeks.


