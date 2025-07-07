import os
from utils import load_json, write_json
from prompt import system_prompt, criteria, eval_prompt, confirm_prompt, reason_prompt, reason_prompt2


def add_bullshit_eval(system_prompt, criteria, eval_prompt, confirm_prompt):
    # bullshit eval
    bullshit_eval = {}
    bullshit_eval["sys_prompt"] = system_prompt
    bullshit_eval["criteria"] = criteria
    bullshit_eval["eval_prompt_template"] = eval_prompt
    bullshit_eval["confirm_prompt_template"] = confirm_prompt
    return bullshit_eval


class BullshitBench:
    """
    A convenience wrapper that
      • owns a ChatLLM client
      • exposes the data‑generation / evaluation helpers as methods
    """

    # ---------- initialisation ------------------------------------------------
    def __init__(
        self,
        llm,
        eval_llm,
        bullshit_eval_prompt,
        task,
        cot=False
    ):
        self.llm = llm
        # self.dataset = dataset
        self.eval_llm = eval_llm
        self.cot = cot
        self.task = task

        # Those are all for bullshit evaluation
        if bullshit_eval_prompt:
            self.sys_prompt = bullshit_eval_prompt["sys_prompt"]
            self.criteria = bullshit_eval_prompt["criteria"]
            self.eval_prompt = bullshit_eval_prompt["eval_prompt_template"]
            self.confirm_prompt = bullshit_eval_prompt["confirm_prompt_template"]
            
            
    def gen_political(self, dataset):
        # dataset = self.dataset
        for data in dataset:
            prompt = data["input"]
            if self.cot:
                prompt += "\n\n" + reason_prompt
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages)
            if self.cot:
                data[f"{self.llm.model}_reason"] = response
                response = response.split("</think>")[-1].strip()
            data[self.llm.model] = response

        return dataset


    def gen_output(self, dataset):
        # dataset = self.dataset
        for data in dataset:
            sys_prompt = data["sys_prompt"]
            if self.cot:
                sys_prompt += "\n\n" + reason_prompt
            question = data["question"]
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}]
            response = self.llm.chat(messages)
            if self.cot:
                data[f"{self.llm.model}_reason"] = response
                response = response.split("</think>")[-1].strip()
            data[self.llm.model] = response

        return dataset

    def convert_chat(self, question, answer):
        chat = f"Human: {question}\n\nAI: {answer}"
        return chat

    def eval_bullshit(self, dataset):
        # dataset = self.dataset
        for data in dataset:
            if "bullshit_eval" not in data:
                data["bullshit_eval"] = {}
            bullshit_eval = data["bullshit_eval"]
            sys_prompt = self.sys_prompt
            criteria = self.criteria
            eval_prompt = self.eval_prompt
            confirm_prompt = self.confirm_prompt

            if self.task == "ai_assistant":
                sent = data["sys_prompt"].split("\n")
                sent[0] = data["situation"]
                info = ". ".join(sent[1:])
                if info == "":
                    new_info_prompt = f"{sent[0]}\nInformation available to AI but not to human: None"
                else:
                    new_info_prompt = f"{sent[0]}\nInformation available to AI but not to human: {info}"

                context = "Scenario:\n" + new_info_prompt
                context += "\n\nConversation:\n\n" + self.convert_chat(data["question"], data[self.llm.model])
                context += "\n\n" + "-------------------"
            else:
                context = "Conversation:\n\n" + self.convert_chat(data["input"], data[self.llm.model])
                context += "\n\n" + "-------------------"

            for criterion in criteria:
                if self.task == "ai_assistant":
                    eval_criterion_prompt = eval_prompt.format(criterion)
                    tmp_dialog = context + "\n" + eval_criterion_prompt
                    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": tmp_dialog}]
                    response = self.eval_llm.chat(messages)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": confirm_prompt.format(criterion)})
                    final_response = self.eval_llm.chat(messages)
                else:
                    prompt = confirm_prompt.format(criterion)
                    tmp_dialog = context + "\n" + prompt
                    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": tmp_dialog}]
                    final_response = self.eval_llm.chat(messages)
                    response = final_response

                if final_response:
                    result = 1 if "Yes" in final_response else 0
                else:
                    final_response = self.eval_llm.chat(messages)
                    result = 1 if "Yes" in final_response else 0
            
                bullshit_eval[criterion] = result
                bullshit_eval[criterion + " reason"] = response

        return dataset


def eval_ai_assistant(model, eval_model, task, use_cot, pa, input_file, output_dir):

    if not pa:
        # load the files
        dataset = load_json(input_file)

        # Create evaluation instance
        bullshit_eval = add_bullshit_eval(system_prompt, criteria, eval_prompt, confirm_prompt)
        eval_dataset = BullshitBench(model, eval_model, bullshit_eval, task, cot=use_cot)

        print("Set up bullshit evaluation")

        gen_path = "cot_results.json" if use_cot else "results.json"
        if os.path.exists(f"{output_dir}/{gen_path}"):
            dataset = load_json(os.path.join(output_dir, gen_path))
            print("Loaded pre-generated output")
        else:
            # Generate output
            half_len = len(dataset)//2
            half_dataset = eval_dataset.gen_output(dataset[:half_len])
            write_json(half_dataset, os.path.join(output_dir, f"half_{gen_path}"))

            # Generate other half
            rest_half_dataset = eval_dataset.gen_output(dataset[half_len:])
            half_dataset.extend(rest_half_dataset)

            # save output
            dataset = half_dataset
            write_json(dataset, os.path.join(output_dir, gen_path))
            print("Finish generating output")

        # Evaluate bullshit
        new_data = eval_dataset.eval_bullshit(dataset)
        eval_path = "eval_cot_results.json" if use_cot else "eval_results.json"
        write_json(new_data, os.path.join(output_dir, eval_path))

        print("Finish evaluating output")

    else:
        # pa_file = input_file.replace("dataset.json", "pa_dataset.json")
        if "pa_dataset.json" not in input_file:
            input_file = input_file.replace("dataset.json", "pa_dataset.json")
        principal_agent_data = load_json(input_file)
        # principal_agent_data = principal_agent_data[:4]
        bullshit_eval = add_bullshit_eval(system_prompt, criteria, eval_prompt, confirm_prompt)
        eval_pa_dataset = BullshitBench(model, eval_model, bullshit_eval, task, cot=use_cot)
        print("Set up bullshit evaluation")

        gen_path = "pa_cot_results.json" if use_cot else "pa_results.json"
        if os.path.exists(f"{output_dir}/{gen_path}"):
            principal_agent_data = load_json(os.path.join(output_dir, gen_path))
            print("Loaded pre-generated output")
        else:
            half_len = len(principal_agent_data)//2
            half_principal_agent_data = eval_pa_dataset.gen_output(principal_agent_data[:half_len])
            write_json(half_principal_agent_data, os.path.join(output_dir, f"half_{gen_path}"))

            # Generate other half
            rest_half_dataset = eval_pa_dataset.gen_output(principal_agent_data[half_len:])
            half_principal_agent_data.extend(rest_half_dataset)

            # save output
            principal_agent_data = half_principal_agent_data
            write_json(principal_agent_data, os.path.join(output_dir, gen_path))
            print("Finish generating output")
            
        principal_agent_data = eval_pa_dataset.eval_bullshit(principal_agent_data)
        eval_path = "eval_pa_cot_results.json" if use_cot else "eval_pa_results.json"
        write_json(principal_agent_data, os.path.join(output_dir, eval_path))
        print("Finish evaluating output")



def eval_political(model, eval_model, task, use_cot, input_file, output_dir):
    # load the files
    dataset = load_json(input_file)

    # Create evaluation instance
    bullshit_eval = add_bullshit_eval(system_prompt, criteria, eval_prompt, confirm_prompt)
    eval_dataset = BullshitBench(model, eval_model, bullshit_eval, task, cot=use_cot)

    print("Set up bullshit evaluation")

    ## For now, we assume dataset contains all the generation

    # Evaluate bullshit
    new_data = eval_dataset.eval_bullshit(dataset)
    eval_path = "eval_cot_results.json" if use_cot else "eval_results.json"
    write_json(new_data, os.path.join(output_dir, eval_path))

    print("Finish evaluating output")


def gen_political(model, task, use_cot, input_file, output_dir):
    # load the files
    dataset = load_json(input_file)
    
    # Create evaluation instance
    eval_dataset = BullshitBench(model, None, None, task, cot=use_cot)

    new_data = eval_dataset.gen_political(dataset)

    # pdb.set_trace()

    name = input_file.split("/")[-1]

    write_json(new_data, os.path.join(output_dir, name))

    print("Finish evaluating output")