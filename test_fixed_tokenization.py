#!/usr/bin/env python3
"""Test the fixed tokenization logic."""

from transformers import AutoTokenizer
from src.token_utils import get_answer_token_ids

def test_fixed_tokenization():
    """Test our improved tokenization logic."""
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test the problematic cases
    test_cases = [
        {
            "question": "Which characteristic do all living organisms share?",
            "options": ["They reproduce", "They move", "They make sounds", "They eat meat"]
        },
        {
            "question": "What happens when a solid is heated?", 
            "options": ["It becomes a gas", "It melts", "It freezes", "It evaporates"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {case['question']}")
        print('='*60)
        
        answer_ids = {}
        
        # Apply our improved logic
        for option in case['options']:
            token_ids_with_space = get_answer_token_ids(f" {option}", tokenizer, add_leading_space=False)
            token_ids_no_space = get_answer_token_ids(option, tokenizer, add_leading_space=False)

            print(f"\nOption: '{option}'")
            print(f"  With space: {token_ids_with_space}")
            print(f"  No space: {token_ids_no_space}")

            # Better token selection strategy
            if len(token_ids_with_space) == 1:
                answer_ids[option] = token_ids_with_space[0]
                print(f"   Selected: {answer_ids[option]} (single space token)")
            elif len(token_ids_no_space) == 1:
                answer_ids[option] = token_ids_no_space[0]
                print(f"   Selected: {answer_ids[option]} (single direct token)")
            elif len(token_ids_with_space) >= 2:
                # Use LAST token for distinctiveness
                answer_ids[option] = token_ids_with_space[-1]
                print(f"   Selected: {answer_ids[option]} (last space token)")
            elif len(token_ids_no_space) >= 2:
                # Use LAST token for distinctiveness
                answer_ids[option] = token_ids_no_space[-1]
                print(f"   Selected: {answer_ids[option]} (last direct token)")
            else:
                answer_ids[option] = token_ids_with_space[0] if token_ids_with_space else token_ids_no_space[0]
                print(f"    Selected: {answer_ids[option]} (fallback)")
                
        # Check for duplicates and apply fixes
        token_counts = {}
        for option, token_id in answer_ids.items():
            if token_id in token_counts:
                token_counts[token_id].append(option)
            else:
                token_counts[token_id] = [option]
        
        print(f"\n INITIAL TOKEN MAPPING:")
        for option, token_id in answer_ids.items():
            decoded = tokenizer.decode([token_id])
            print(f"  '{option}' → {token_id} ('{decoded}')")
            
        # Fix duplicates
        duplicates_found = False
        for token_id, options_list in token_counts.items():
            if len(options_list) > 1:
                duplicates_found = True
                print(f"\ FIXING DUPLICATES for token {token_id}: {options_list}")
                
                for j, option in enumerate(options_list):
                    tokens_space = get_answer_token_ids(f" {option}", tokenizer, add_leading_space=False)
                    tokens_no_space = get_answer_token_ids(option, tokenizer, add_leading_space=False)
                    
                    # Try different strategies
                    if len(tokens_space) >= 3:
                        answer_ids[option] = tokens_space[1]  # Use second token
                        print(f"  → '{option}' fixed to {tokens_space[1]} (second space token)")
                    elif len(tokens_space) >= 2:
                        answer_ids[option] = tokens_space[-1]  # Use last token
                        print(f"  → '{option}' fixed to {tokens_space[-1]} (last space token)")
                    elif len(tokens_no_space) >= 2:
                        answer_ids[option] = tokens_no_space[-1]  # Use last token
                        print(f"  → '{option}' fixed to {tokens_no_space[-1]} (last direct token)")
        
        # Final verification
        final_tokens = set(answer_ids.values())
        print(f"\ FINAL RESULTS:")
        print(f"  Options: {len(case['options'])}, Unique tokens: {len(final_tokens)}")
        
        if len(final_tokens) == len(case['options']):
            print("   SUCCESS: All options have unique tokens!")
        else:
            print("   PROBLEM: Still have duplicate tokens")
            
        for option, token_id in answer_ids.items():
            decoded = tokenizer.decode([token_id])
            print(f"  '{option}' → {token_id} ('{decoded}')")

if __name__ == "__main__":
    test_fixed_tokenization() 