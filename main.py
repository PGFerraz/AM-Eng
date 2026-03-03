"""
AM_Eng Created by: Pedro Gabriel Ferraz Santos Silva
https://github.com/PGFerraz

This is Applied Machine Engineering. A command-line chat interface for a llm, desinigned to ask questions 
about computer science and engineering. It was created and trained with a standard GPU-equipped (6GB VRAM 
or more) home computer in mind. In the future it will be trained with aditional databases,for a better 
comprehension of the subjects. I will also comment every step of the code, so it can be easily 
understood and modified by anyone.
"""

import os
import json
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import logging as transformers_logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setting up environment variables and logging 
""" 
First line silents the inrrelevant transformes library warnings, so it only shows errors; 

Second line disables the parallelism of the tokenizers library, which can cause issues in some environments; 
"""
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Defining constants for the llm memory archive and the model name. 
# The memory archive is a json file where the llm will store facts and information
MEMORY_UNIT = "fact_unit.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# This funcion is responsible for print the llm output in a typewitter style.
def type_print(text, prefix="\033[91mAM:\033[0m "):
    print(prefix, end="")
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.005)
    print("\n")


# These two functions are responsible for loading and saving the facts in the memory archive.
def load_facts():
    if not os.path.exists(MEMORY_UNIT):
        return []
    try:
        with open(MEMORY_UNIT, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"\033[31m[Memory Load Error]: {e}\033[0m")
        return []

def save_facts(raw_output):
    # If needed, you can uncomment
    # print(f"\033[93m[DEBUG Extrator]: {repr(raw_output)}\033[0m")

    try:
        parsed = json.loads(raw_output.strip())

        if not parsed:
            return

        if isinstance(parsed, dict) and "fact" in parsed:
            fact_text = parsed["fact"].strip()

            #   Dinamic blocking of non-persistent patterns to avoid memory pollution.
            blocked_patterns = [
                "asked",
                "question",
                "requested",
                "wants to know",
            ]

            if any(p in fact_text.lower() for p in blocked_patterns):
                return

            existing_data = load_facts()

            if fact_text not in [f.get("fact") for f in existing_data]:
                existing_data.append({"fact": fact_text})

                with open(MEMORY_UNIT, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

    except json.JSONDecodeError:
        print("\033[31m[Extractor Error: Invalid JSON]\033[0m")


# Just a print to show that the llm is initializing.
print("\033[93m[AM_Eng Initializing...]\033[0m")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# Defining the model and tokenizer. 
"""
'AutoModelForCasualLM' is a class from the Hugging Face Transformers library that allows us to load a 
pre-trained language model for causal language modeling tasks; 

'model_name' is the constant defined above; 

'device_map="auto"' allows the library to automatically decide whether to use CPU or GPU for inference, 
based on the available hardware. It uses the accelerate library; 

'torch_dtype=torch.float16' specifies that the model should be loaded using 16-bit floating point precision, 
which can reduce memory usage and speed up inference on compatible hardware. 
"""
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)
# 'AutoTokenizer' is a class from the Hugging Face Transformers library that allows 
# us to load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Initializing the chat context, which is a list that will store the current conversation history.
chat_context = []

os.system("clear" if os.name == "posix" else "cls")

# Just a print to show that the llm is ready to receive input.
print("\033[91mAM_Eng — Your CS & Engineering Assistant.\033[0m\n")

# Main loop of the program. It will keep running until the user types 'quit', 'exit' or '/clear'.

def main():
    while True:

        # Getting user input. The input is wrapped in a try-except block to catch the KeyboardInterrupt exception, 
        # which is raised when the user presses Ctrl+C. This allows us to exit the program
        try:
            prompt = input("\033[92mINPUT>\033[0m ").strip()
            if not prompt:
                continue
        except KeyboardInterrupt:
            break
        
        # Checking if the user wants to quit, exit or clear the console. If the user types '/clear', 
        # the console will be cleared and the loop will continue.
        if prompt.lower() in ["quit", "exit"]:
            break
        if prompt.lower() == "/clear":
            os.system("clear" if os.name == "posix" else "cls")
            chat_context = []
            continue

        """    
        Loading past facts from the memory archive. If there are any past facts, they will be formatted 
        and added to the system prompt as context for the llm. This allows the llm to use previously 
        extracted facts about the user in its responses, making the conversation more personalized and 
        coherent over time. Only the last 5 facts are included to keep the context relevant and concise. 
        If there are no past facts, the memory context will be empty.
        """
        past_facts = load_facts()

        if past_facts:
            facts_text = "\n".join([f"- {fact['fact']}" for fact in past_facts[-5:]])
            memory_context = f"\nKnown persistent user facts:\n{facts_text}\n"
        else:
            memory_context = ""

        # Defining the system prompt for the llm (behavior, style and instructions).
        system_content = (
            "You are a pragmatic senior systems engineer.\n"
            "Be technical but natural.\n"
            "Be concise but not robotic.\n"
            "Use known persistent user facts when relevant.\n"
            f"{memory_context}"
        )

        # Building the messages list, which is the input for the llm. It consists of the system prompt,
        # and the last 4 messages of the chat context.
        messages = [{"role": "system", "content": system_content}]
        messages.extend(chat_context[-4:])
        messages.append({"role": "user", "content": prompt})

        """
        This is the first step of the llm response generation. The llm will first try to extract any stable and
        persistent facts about the user from the current prompt. The llm will then generate a response based 
        on this prompt, and the output will be parsed and saved in the memory archive if it contains valid facts. 
        This allows the llm to build a long-term memory of the user over time,
        """
        # The 'fact_messages' list is a separate input for the llm, designed to extract facts from the user prompt.
        fact_messages = [
            {
                "role": "system",
                "content": (
                    "You extract long-term persistent facts about the USER only.\n"
                    "Only extract stable information like name, skills, preferences, projects, or location.\n"
                    "Do NOT extract questions or temporary actions.\n"
                    "Return strictly valid JSON like: {\"fact\": \"User ...\"}\n"
                    "If no stable personal fact is present, return {} ONLY."
                )
            },
            {
                "role": "user",
                "content": f"The USER said: {prompt}"
            }
        ]

        # 'fact_input' is created by applying the chat template to the 'fact_messages' list.
        # 'tokenize=False' argument means that the messages will not be tokenized at this stage
        # 'add_generation_prompt=True' argument adds a special token to indicate where the model should start generating output.
        fact_input = tokenizer.apply_chat_template(
            fact_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 'f_inputs' is the tokenized version of 'fact_input'
        f_inputs = tokenizer(fact_input, return_tensors="pt").to(model.device)

        # The inference is wrapped in 'torch.inference_mode()' context, which disables gradient calculations and other training-related features.
        with torch.inference_mode():
            # 'f_outputs' is the output of the model when given 'f_inputs'. 
            # The generation parameters are set to produce a deterministic output (do_sample=False)
            # maximum of 60 new tokens.
            f_outputs = model.generate(
                **f_inputs,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True
            )  

            # The generated output is decoded back into text, skipping special tokens.
            extracted = tokenizer.decode(
                f_outputs[0][f_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # saving the extracted facts in the memory archive using the 'save_facts' function defined above.
            save_facts(extracted)

            # Main response generation step. 
            # The 'messages' list is converted into a prompt using the chat template, and then tokenized.
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 'm_inputs' is the tokenized version of 'full_prompt', ready to be fed into the model for response generation.
            m_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            # The model generates a response based on 'm_inputs'.
            m_outputs = model.generate(
                **m_inputs,
                max_new_tokens=600,
                do_sample=True,
                use_cache=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        # The generated response is decoded back into text, skipping special tokens and stripping any leading/trailing whitespace.
        response = tokenizer.decode(
            m_outputs[0][m_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        # The response is printed to the console using the 'type_print' function defined above.
        type_print(response)

        # Finally, the current user prompt and the llm response are appended to the 'chat_context' list.
        chat_context.append({"role": "user", "content": prompt})
        chat_context.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
