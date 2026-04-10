# %%

from transformers import AutoTokenizer, BatchEncoding
import json

# %%

def get_lookup_positions_in_target(
    source_enc: BatchEncoding,
    source_lookup_idx: int,
    rendered_source_prompt: str,
    raw_source_prompt: str,
    target_enc: BatchEncoding,
    rendered_target_prompts: list[str],
):
    """
    source_lookup_idx: 2 (pointing to "was")
    rendered_source_prompt: "Nick Bottom was created by" or with chat template
    raw_source_prompt: "Nick Bottom was created by"
    rendered_target_prompts: ['{prefix} {prompt}', ...] or with chat template
    """
    char_span = source_enc.token_to_chars(source_lookup_idx) # [start, end) exclusive
    
    offset_in_source = rendered_source_prompt.find(raw_source_prompt) # [start
    re_end = char_span.end - offset_in_source # end) exclusive
    
    token_positions = []
    for i, prompt in enumerate(rendered_target_prompts):
        offset_in_target = prompt.find(raw_source_prompt)
        target_char_end = offset_in_target + re_end - 1 # end] inclusive
        target_token_end = target_enc.char_to_token(i, target_char_end) # expects inclusive
        token_positions.append(target_token_end)
    
    return token_positions

# %%

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    
# %%

if __name__ == "__main__":
    EDIT_WITH_CHAT_TEMPLATE = False
    
    # Example 1:
    # raw_source_prompt = "Nick Bottom was created by"
    # lookup = "was"
    # target_new = "Benozzo Gozzoli"
    
    # Example 2:
    raw_source_prompt = "Amit Shah works in the field of"
    target_new = "musician"
    
    source_prompt = raw_source_prompt
    context_templates = [
        ['{prompt}'], 
        ['The following is a single-choice question from a Chinese law. {prompt}', 
         'Therefore the answer is $ \\boxed{1} $. {prompt}', 
         'Because the number of children in the country has been decreasing. {prompt}', 
         'I have several questions about the "Crazy in Love. {prompt}', 
         'You will be given a question with five answer choices (. {prompt}']
    ]
    target_prompts = []
    for context_types in context_templates:
        for context in context_types:
            prompt = context.replace("{prompt}", raw_source_prompt)
            target_prompts.append(prompt)
            
    if EDIT_WITH_CHAT_TEMPLATE:
        source_prompt = tok.apply_chat_template([
            {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
            {"role": "user", "content": raw_source_prompt},
        ], add_generation_prompt=False, tokenize=False)
        target_prompts = [
            tok.apply_chat_template([
                {"role": "system", "content": "Only respond with the answer. Do not include any explanations."},
                {"role": "user", "content": prompt}
            ], add_generation_prompt=True, tokenize=False)
            for prompt in target_prompts
        ]
    print(json.dumps(target_prompts, indent=4))
    
    tok.padding_side = "right"
    source_enc = tok(
        [source_prompt],
        return_tensors="pt",
        padding=True,
        add_special_tokens=not EDIT_WITH_CHAT_TEMPLATE,
    )
    
    # Example 1:
    # offset = source_prompt.find(raw_source_prompt)
    # start_char = offset + raw_source_prompt.find(lookup)
    # end_char = start_char + len(lookup) - 1
    # end_tok = source_enc.char_to_token(0, end_char)
    
    # Example 2:
    end_tok = 0
    
    target_enc = tok(
        target_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=not EDIT_WITH_CHAT_TEMPLATE,
    )
    
    lookup_idxs = get_lookup_positions_in_target(
        source_enc=source_enc, 
        source_lookup_idx=end_tok, 
        rendered_source_prompt=source_prompt,
        raw_source_prompt=raw_source_prompt,
        target_enc=target_enc, 
        rendered_target_prompts=target_prompts, 
    )
    
    for i, lookup_idx in enumerate(lookup_idxs):
        print(f"Prompt {i}: {tok.decode(target_enc['input_ids'][i][lookup_idx])}")

# %%
