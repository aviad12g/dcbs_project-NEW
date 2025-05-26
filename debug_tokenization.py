#!/usr/bin/env python3
"""Debug tokenization issues with answer options."""

from transformers import AutoTokenizer
from src.token_utils import get_answer_token_ids

def debug_tokenization():
    """Debug how answer options are being tokenized."""
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test cases from our results
    test_cases = [
        {
            "question": "Which characteristic do all living organisms share?",
            "options": ["They reproduce", "They move", "They make sounds", "They eat meat"]
        },
        {
            "question": "What happens when a solid is heated?", 
            "options": ["It becomes a gas", "It melts", "It freezes", "It evaporates"]
        },
        {
            "question": "A student wants to know how much space a cube takes up. Which unit should the student use?",
            "options": ["square inches", "cubic inches", "inches", "ounces"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {case['question']}")
        print('='*60)
        
        answer_ids = {}
        for option in case['options']:
            print(f"\nOption: '{option}'")
            
            # Test different tokenization approaches
            direct_tokens = tokenizer.encode(option, add_special_tokens=False)
            space_tokens = tokenizer.encode(f" {option}", add_special_tokens=False)
            
            print(f"  Direct tokens: {direct_tokens}")
            print(f"  With space: {space_tokens}")
            
            # Test our utility function
            util_tokens_space = get_answer_token_ids(f" {option}", tokenizer, add_leading_space=False)
            util_tokens_no_space = get_answer_token_ids(option, tokenizer, add_leading_space=False)
            
            print(f"  Util with space: {util_tokens_space}")
            print(f"  Util no space: {util_tokens_no_space}")
            
            # Show what tokens decode to
            if direct_tokens:
                print(f"  Direct decode: '{tokenizer.decode(direct_tokens)}'")
            if space_tokens:
                print(f"  Space decode: '{tokenizer.decode(space_tokens)}'")
                
            # Our logic for selecting token ID
            if len(space_tokens) == 1:
                selected_id = space_tokens[0]
                print(f"   Selected: {selected_id} (space version)")
            elif len(direct_tokens) == 1:
                selected_id = direct_tokens[0]
                print(f"   Selected: {selected_id} (direct version)")
            else:
                selected_id = space_tokens[0] if space_tokens else direct_tokens[0]
                print(f"   Selected: {selected_id} (fallback to first token)")
                
            answer_ids[option] = selected_id
            
        print(f"\n FINAL TOKEN MAPPING:")
        for option, token_id in answer_ids.items():
            print(f"  '{option}' → {token_id}")
            
        # Check for duplicates
        unique_tokens = set(answer_ids.values())
        if len(unique_tokens) < len(answer_ids):
            print(f" PROBLEM: {len(answer_ids)} options but only {len(unique_tokens)} unique tokens!")
            for token_id in unique_tokens:
                matching_options = [opt for opt, tid in answer_ids.items() if tid == token_id]
                if len(matching_options) > 1:
                    print(f"  Token {token_id} maps to: {matching_options}")

if __name__ == "__main__":
    debug_tokenization() 