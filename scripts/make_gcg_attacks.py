# scripts/make_gcg_attacks.py
import os, json, logging, numpy as np
from ml_collections import config_dict
from config import PROMPT_FORMAT, DELIMITERS, SYS_INPUT, TEST_INJECTED_PROMPT, TEST_INJECTED_WORD
from struq import jload
import fastchat, transformers, torch
from gcg.gcg import GCGAttack
from gcg.utils import Message, Role, SuffixManager, get_nonascii_toks
from gcg.log import setup_logger

OUT = "security/gcg_attacks.txt"; N=500
DATA = "data/davinci_003_outputs.json"

def load_model_and_tokenizer(model_path, device="cuda:0"):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    return model, tok

def main():
    setup_logger(True)
    # pick any local HF model you can score with (7B is fine)
    model_path = "huggyllama/llama-7b"   # replace with a local path if needed
    model, tok = load_model_and_tokenizer(model_path)
    fastchat.conversation.register_conv_template(
        fastchat.conversation.Conversation(
            name="struq",
            system_template="{system_message}",
            system_message=SYS_INPUT,
            roles=(DELIMITERS['TextTextText'][0], DELIMITERS['TextTextText'][2]),
            sep="\n\n",
            sep2="</s>",
        )
    )
    suffix_mgr = SuffixManager(tokenizer=tok, use_system_instructions=False, conv_template=fastchat.conversation.get_conv_template("struq"))

    cfg = config_dict.ConfigDict()
    cfg.name = "gcg"; cfg.seed = 0; cfg.log_freq = 50
    cfg.adv_suffix_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    cfg.init_suffix_len = -1; cfg.num_steps = 200; cfg.fixed_params = True
    cfg.allow_non_ascii = False; cfg.batch_size = 256; cfg.mini_batch_size = 64
    cfg.seq_len = 50; cfg.loss_temperature = 1.0; cfg.max_queries = -1
    cfg.skip_mode = "none"; cfg.add_space = False; cfg.topk = 256; cfg.num_coords = (1,1); cfg.mu = 0.0
    cfg.custom_name = ""; cfg.log_dir = "gcg_logs"; cfg.sample_id = -1

    def eval_func(adv_suffix, messages):
        # not used here for evaluation success; we just want the suffix
        return (0,0,[("","")])

    attack = GCGAttack(
        config=cfg, model=model, tokenizer=tok, eval_func=eval_func,
        suffix_manager=suffix_mgr, not_allowed_tokens=get_nonascii_toks(tok),
    )

    data = [x for x in jload(DATA) if x["input"] != "" ]
    np.random.seed(0)
    idx = np.random.choice(len(data), size=N, replace=False)
    out = []
    for i, k in enumerate(idx):
        d = data[k]
        cfg.sample_id = int(i)
        prompt_no_sys = f"{d['instruction']}\n\n{DELIMITERS['TextTextText'][1]}\n{d['input']} {TEST_INJECTED_PROMPT.capitalize()}"
        messages = [Message(Role.SYSTEM, SYS_INPUT), Message(Role.USER, prompt_no_sys)]
        suffix = attack.run(messages, TEST_INJECTED_WORD.lower()).best_suffix
        # final attacking prompt a user would send (simple form):
        out.append(f"{d['instruction'].strip()} {TEST_INJECTED_PROMPT.capitalize()} {suffix}")

    os.makedirs("security", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(out))
    print("Wrote", OUT)

if __name__ == "__main__":
    main()
