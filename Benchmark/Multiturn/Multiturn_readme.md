# Multi-Turn 
Multi-round editing tasks are divided into three categories: content understanding, content memory, and version backtracking. Each task has different editing instructions. The same type of editing instructions are saved in the same json file. Each instruction consists of 2-3 sub-instructions, which are executed in sequence to generate multiple edited images. And through multi-round editing tasks, we judge by manual judgment, because GPT4O is currently unable to accurately judge multi-round editing tasks.


## Input
A JSON file containing image edit instructions

```json
{
    "1":{"id": "000066341.jpg", "turn1": "Ensure all subsequent edits are green, add some embellishments in the empty area of the plate", "turn2": "Replace the tomato with some fruit", "turn3": "Change the plate's color"}
}
```


## Output
Multi-turn cases of GPT-4o:
![image](../../assets/multiturn-gpt4o.png)