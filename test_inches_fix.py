#!/usr/bin/env python3
"""Test the inches tokenization fix."""

from transformers import AutoTokenizer

def test_inches_fix():
    """Test improved tokenization for the inches case."""
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    options = ["square inches", "cubic inches", "inches", "ounces"]
    
    print("🔍 IMPROVED TOKENIZATION LOGIC:")
    print("="*50)
    
    answer_ids = {}
    
    for option in options:
        token_ids_with_space = tokenizer.encode(f" {option}", add_special_tokens=False)
        token_ids_no_space = tokenizer.encode(option, add_special_tokens=False)
        
        print(f"\nOption: '{option}'")
        print(f"  With space: {token_ids_with_space}")
        print(f"  No space: {token_ids_no_space}")
        
        # Apply improved logic
        if len(token_ids_with_space) == 1:
            answer_ids[option] = token_ids_with_space[0]
            print(f"   Selected: {answer_ids[option]} (single space token)")
        elif len(token_ids_no_space) == 1:
            answer_ids[option] = token_ids_no_space[0]
            print(f"   Selected: {answer_ids[option]} (single direct token)")
        elif len(token_ids_with_space) >= 2:
            # Use first token (more distinctive)
            first_token = token_ids_with_space[0]
            decoded_first = tokenizer.decode([first_token]).strip()
            
            # If first token is very short or common, use second token
            if len(decoded_first) <= 2 or decoded_first.lower() in [' the', ' a', ' an', ' it', ' they']:
                answer_ids[option] = token_ids_with_space[1] if len(token_ids_with_space) > 1 else first_token
                print(f"   Selected: {answer_ids[option]} (second space token, first was common)")
            else:
                answer_ids[option] = first_token
                print(f"   Selected: {answer_ids[option]} (first space token)")
        elif len(token_ids_no_space) >= 2:
            # Same logic for no-space tokens
            first_token = token_ids_no_space[0]
            decoded_first = tokenizer.decode([first_token]).strip()
            
            if len(decoded_first) <= 2 or decoded_first.lower() in ['the', 'a', 'an', 'it', 'they']:
                answer_ids[option] = token_ids_no_space[1] if len(token_ids_no_space) > 1 else first_token
                print(f"   Selected: {answer_ids[option]} (second direct token, first was common)")
            else:
                answer_ids[option] = first_token
                print(f"   Selected: {answer_ids[option]} (first direct token)")
        else:
            # Fallback
            answer_ids[option] = token_ids_with_space[0] if token_ids_with_space else token_ids_no_space[0]
            print(f"    Selected: {answer_ids[option]} (fallback)")
    
    print(f"\n FINAL RESULTS:")
    print("="*50)
    
    for option, token_id in answer_ids.items():
        decoded = tokenizer.decode([token_id])
        print(f"  '{option}' → {token_id} ('{decoded}')")
    
    # Check uniqueness
    unique_tokens = set(answer_ids.values())
    print(f"\n Unique tokens: {len(unique_tokens)} out of {len(options)} options")
    
    if len(unique_tokens) == len(options):
        print(" SUCCESS: All options have unique tokens!")
    else:
        print(" PROBLEM: Still have duplicate tokens")
        # Show duplicates
        for token_id in unique_tokens:
            matching_options = [opt for opt, tid in answer_ids.items() if tid == token_id]
            if len(matching_options) > 1:
                print(f"  Token {token_id} maps to: {matching_options}")

if __name__ == "__main__":
    test_inches_fix() 