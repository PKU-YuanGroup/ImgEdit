import random
import torch
from concept.utils import process_text_multi_turn
import spacy

'''
need_summarize : object num, object space percentage, 

'''


edit_type = {
    "add", 
    "add_wo_ambiguity",
    "remove",
    "remove_wo_ambiguity",
    "replace",
    "replace_wo_ambiguity",
    "edit",
    # "edit":["color_alter", "material_alter", "texture_alter", "appearance_alter", "with reference", "without reference"],
    "background_change",
    "textual",
}

feature_in_multi_turn = {
    "not_specify_subject",
    "edit the same class but not same object",
    "use other way to discribe subject",
    "not_specify_edit_type",
    "revoke-to-turn-n",
    "denial_and_retry(back to last modify)",
}

# downstream_task = {
#     "bbox",
#     "generate mask",
#     "scribble",
#     "depth",
#     "scratch",
# }
# gpt4o = {
#     "wise-like implicit edit"
#     "position exchange"
#     "split object"
#     "merge object"
# }

# available_dataset = {
#     "multi-view",["stock data", "MVImgNet"]
#     "action_change", ["video dataset"]
# }

# selected_from_video = {
#     "add with" ["id_consistency"]
# }

'''
when editing original image in all round, only need to take care about object(same or identical)
'''

def get_content_instruction(new_prompt, few_shot_examples, instruction_type, tokenizer):
    system_message = f"You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'edited object','output'. Here is the latest conversation between Assistant and User."
    if instruction_type == 'add':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for adding an object to the image ('edit'), the object to add ('edited object'), and the caption with the object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {{'edit': '<instruction>', 'edited object': '<object>', 'output': '<caption>'}}. Construct the instruction with one of the following instruction words: ['place', 'add', 'include']. Don't include any \" or edit any actions in the instruction.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'remove':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for removing an object to the image ('edit'), the object to remove ('edited object'), and the caption with the object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['erase', 'remove', 'delete']. Don't include any \" or edit any actions in the instruction.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'replace':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'edited object', 'new object', 'output'. Notice that the 'edited object' should not be human and nouns about human. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for replacing an object to another object in the image ('edit'), the original object to replace ('edited object'), the new object ('new object'), the caption with the new object ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<object>', 'new object': '<new object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'replace']. Don't include any \" in the instruction and don't use remove instruction. The new object cannot be empty.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, new object, and output caption for you. Let's get started!"]
    elif instruction_type == 'color_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing the color of one object in the image ('edit'), the object to change color ('edited object'), the caption of the object with new color ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'turn']. Use the following format to construct instruction: {change/alter/turn the color of the <object> to <color>}. Don't include any \" in the response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'material_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's material in the image ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. The material should be selected from 'wooden', 'vitreous', 'metallic', 'statuary' and 'paper'. Use the following format to construct instruction: {change/alter/turn/make the material of the <object> to <material>}. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'texture_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's texture in the image ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following texture words: ['dotted', 'striped', 'brushy', 'woven', 'meshed']. Use the following format to construct instruction: {change/alter/turn/make the texture of the <object> to <texture>}. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'appearance_alter':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's appearance in the image, such as texture or material ('edit'), the object to change appearance ('edited object'), the caption with the object of the new appearance ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['turn', 'make']. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'action_change':
        intro_message = [
            "Hi, My job is to take a given caption ('input') and to output the following: an instruction for changing an object's action in the image ('edit'), the object to change action ('edited object'), the caption of object with a new action ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'edited object': '<original object>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['change', 'turn', 'make']. Use the following format to construct instruction: {change/turn/make the action of the <object> to <action>}. Don't include any \" in the instruction and only change the action.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, edited object, and output caption for you. Let's get started!"]
    elif instruction_type == 'background_change':
        # introduction message #
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'new background', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job to take a given caption ('input') and to output the following: an instruction for changing the background in the image ('edit'), the new background ('new background'), the caption with the new background ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'new background': '<background>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['alter', 'change', 'turn']. Use the following format to construct instruction: {change/alter/turn the background to <background>}. The new background should be reasonable with objects. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, the new background, and output caption for you. Let's get started!"]
    elif instruction_type == 'tone_transfer':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'new state', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, My job to take a given caption ('input') and to output the following: an instruction for changing the overall state about weather/time/season in the image ('edit'), the new state ('new state'), the caption with the new state ('output'). Please help me do it. I will give you the 'input', and you will help. When you reply, use the following format: {'edit': '<instruction>', 'new state': '<state>', 'output': '<caption>'}. Construct the instruction with one of the following instruction words: ['make', 'change', 'turn']. Use the following format to construct instruction: {change/make/turn the weather/time/season to <state>}. The new state should only be about time change, weather change, or season change. Don't include any \" in the instruction and response.",
            "Sure, I'd be happy to help! Just provide me with the 'input' (the original caption), and I'll generate the instruction, the new state, and output caption for you. Let's get started!"]
    elif instruction_type == 'textual':
        system_message = "You are an assistant that only speaks JSON. Do not write normal text. The assistant answer is JSON with the following string fields: 'edit', 'input', 'output'. Here is the latest conversation between Assistant and User."
        intro_message = [
            "Hi, my job is to modify a given description and text and output the following: an instruction for changing the text ('edit'), the combined description and text as input ('input'), and the modified version as output ('output'). Please help me do it. When you reply, use the following format: {'edit': '<instruction>', 'input': '<caption>', 'output': '<caption>'}. Construct the instruction using one of the following words: ['alter', 'change', 'replace', 'turn'] . You should ensure the number of words of the text remains the same before and after the change.",
            "Sure, I'd be happy to help! Just provide me with a 'description' and 'text', and I'll generate the instruction, input, and output for you. Let's get started!"
        ]
    elif instruction_type == 'visual_reference':
        # 不是直接产生相关指令，相关指令用remove和replace来转化，输入的input是现在这个的output
        # input: a cow is sitting on a grass ; output: a cow and a rabbit are sitting on a grass
        # visual reference的input和output都是自己产生的，input保留一个object在一个场景，output加上一个新的合理object（比较大的对象）
        return
    else:
        print(f'Error Type {instruction_type}')
        raise NotImplementedError
    # shuffling #
    random.seed(torch.randint(1 << 32, ()).item())
    random.shuffle(few_shot_examples) #  shuffling between examples
    #  few_shot_examples: randomly sampling 60% of the examples
    examples = []
    for e in few_shot_examples[:5]:
        examples += e
    prompt = process_text_multi_turn(history=intro_message+examples, text=new_prompt,
                                     tokenizer=tokenizer, system_prompt=system_message)

    return prompt


def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns':[], 'adj':[], 'verb': []}
    words_list = nlp(raw_text)
    for i in range(len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN':
            tags['adj'].append(token.text.lower())
        elif token.pos_ == 'NOUN' and next_token.pos_ != 'ADP':
            tags['nouns'].append(token.text.lower())
        elif token.pos_ == 'VERB':
            tags['verb'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN':
        tags['nouns'].append(words_list[-1].text.lower())
    return tags