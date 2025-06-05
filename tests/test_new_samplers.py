import pytest
import torch
import numpy as np
from src.dcbs.samplers.temperature_sampler import TemperatureSampler
from src.dcbs.samplers.top_k_sampler import TopKSampler
from src.dcbs.samplers.greedy_sampler import GreedySampler # For comparison

# Helper function to set random seed for reproducibility in tests
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TestTemperatureSampler:

    def setup_method(self):
        set_seed(42)
        self.vocab_size = 100
        self.logits = torch.randn(self.vocab_size)

    def test_init_valid_temperature(self):
        sampler = TemperatureSampler(temperature=0.5)
        assert sampler.temperature == 0.5
        sampler = TemperatureSampler(temperature=1.0)
        assert sampler.temperature == 1.0
        sampler = TemperatureSampler(temperature=2.0)
        assert sampler.temperature == 2.0

    def test_init_invalid_temperature_zero(self):
        with pytest.raises(ValueError, match="Temperature must be a positive float."):
            TemperatureSampler(temperature=0.0)

    def test_init_invalid_temperature_negative(self):
        with pytest.raises(ValueError, match="Temperature must be a positive float."):
            TemperatureSampler(temperature=-0.1)

    def test_sample_default_temperature_1_0(self):
        sampler = TemperatureSampler(temperature=1.0)
        selected_token = sampler.sample(self.logits)
        # With temperature 1.0, it's standard softmax sampling
        # Cannot assert exact token due to randomness, but can check type and range
        assert isinstance(selected_token, int)
        assert 0 <= selected_token < self.vocab_size

    def test_sample_low_temperature_approaches_greedy(self):
        sampler = TemperatureSampler(temperature=0.01) # Very low temperature
        selected_token = sampler.sample(self.logits)
        greedy_token = torch.argmax(self.logits).item()
        # With very low temperature, it should almost always pick the greedy token
        # Use a loop to account for very small chance of deviation due to multinomial
        num_trials = 100
        greedy_matches = 0
        for _ in range(num_trials):
            if sampler.sample(self.logits) == greedy_token:
                greedy_matches += 1
        assert greedy_matches >= num_trials * 0.99 # Expect very high match rate

    def test_sample_high_temperature_approaches_random(self):
        sampler = TemperatureSampler(temperature=100.0) # Very high temperature
        selected_token = sampler.sample(self.logits)
        # With very high temperature, distribution is flattened, more uniform
        # Cannot assert exact token, but test against greedy to ensure it's NOT greedy
        greedy_token = torch.argmax(self.logits).item()
        # Run multiple times to see if it deviates from greedy
        num_trials = 100
        greedy_matches = 0
        for _ in range(num_trials):
            if sampler.sample(self.logits) == greedy_token:
                greedy_matches += 1
        assert greedy_matches < num_trials * 0.5 # Expect less than 50% match rate

    def test_sample_with_filter_tokens(self):
        filter_tokens = {10, 20, 30, 40, 50} # A subset of allowed tokens
        sampler = TemperatureSampler(temperature=1.0)
        # Ensure selected token is from the filter_tokens set
        selected_token = sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert selected_token in filter_tokens

        # Test with filter_tokens where best token is outside filter
        # Create logits where the best token is NOT in filter_tokens
        new_logits = torch.randn(self.vocab_size)
        new_logits[0] = 100.0 # Make 0 the greedy token
        filter_tokens_excluding_0 = {1, 2, 3}
        selected_token = sampler.sample(new_logits, filter_tokens=filter_tokens_excluding_0)
        assert selected_token in filter_tokens_excluding_0

    def test_sample_with_empty_filter_tokens(self):
        sampler = TemperatureSampler(temperature=1.0)
        # If filter_tokens is empty, it should effectively act as if no tokens are allowed
        # The current implementation will set all logits to -inf then normalize, resulting in uniform over all.
        # This might need clarification on expected behavior if an empty filter_tokens means *no* tokens allowed.
        # For now, we expect it to still return a token, as the probabilities become uniform.
        selected_token = sampler.sample(self.logits, filter_tokens=set())
        assert isinstance(selected_token, int)
        assert 0 <= selected_token < self.vocab_size

    def test_sample_with_filter_tokens_no_overlap(self):
        # Test a case where filter_tokens leads to no valid options with very low temperature
        sampler = TemperatureSampler(temperature=0.01)
        # Make a specific token very high, but exclude it from filter_tokens
        special_logits = torch.randn(self.vocab_size)
        special_logits[5] = 1000.0 # Make token 5 overwhelmingly likely
        filter_tokens = {1, 2, 3, 4} # Token 5 is not in filter_tokens

        # Even with filtering, with very low temperature, it should pick the highest among allowed
        selected_token = sampler.sample(special_logits, filter_tokens=filter_tokens)
        assert selected_token in filter_tokens

        # Verify it picks the highest in the filtered set
        expected_token_in_filtered_set = -1
        max_logit_in_filtered = -float('inf')
        for token_id in filter_tokens:
            if special_logits[token_id].item() > max_logit_in_filtered:
                max_logit_in_filtered = special_logits[token_id].item()
                expected_token_in_filtered_set = token_id

        # Due to multinomial's slight randomness, assert that it's highly likely to be the expected token
        num_trials = 100
        expected_matches = 0
        for _ in range(num_trials):
            if sampler.sample(special_logits, filter_tokens=filter_tokens) == expected_token_in_filtered_set:
                expected_matches += 1
        assert expected_matches >= num_trials * 0.95


class TestTopKSampler:

    def setup_method(self):
        set_seed(42)
        self.vocab_size = 100
        # Create logits with distinct values for easier top-k verification
        self.logits = torch.arange(self.vocab_size).float()
        # Shuffle for more realistic distribution but still predictable top-k
        perm = torch.randperm(self.vocab_size)
        self.logits = self.logits[perm]
        self.sorted_logits, self.sorted_indices = torch.sort(self.logits, descending=True)

    def test_init_valid_k(self):
        sampler = TopKSampler(k=1)
        assert sampler.k == 1
        sampler = TopKSampler(k=50)
        assert sampler.k == 50

    def test_init_invalid_k_zero(self):
        with pytest.raises(ValueError, match="Top-K \\(k\\) must be a positive integer."):
            TopKSampler(k=0)

    def test_init_invalid_k_negative(self):
        with pytest.raises(ValueError, match="Top-K \\(k\\) must be a positive integer."):
            TopKSampler(k=-1)

    def test_sample_basic_top_k(self):
        k = 5
        sampler = TopKSampler(k=k)
        selected_token = sampler.sample(self.logits)
        
        # The selected token should be one of the top-k tokens
        top_k_expected_tokens = self.sorted_indices[:k].tolist()
        assert selected_token in top_k_expected_tokens

        # Verify that it samples only from top-k, not just greedy
        greedy_token = self.sorted_indices[0].item()
        num_trials = 100
        non_greedy_samples = 0
        for _ in range(num_trials):
            sample = sampler.sample(self.logits)
            assert sample in top_k_expected_tokens
            if sample != greedy_token:
                non_greedy_samples += 1
        
        # For k > 1, expect some samples to not be the greedy choice (unless probabilities are extremely skewed)
        if k > 1:
            assert non_greedy_samples > 0

    def test_sample_k_equals_vocab_size(self):
        sampler = TopKSampler(k=self.vocab_size)
        selected_token = sampler.sample(self.logits)
        assert 0 <= selected_token < self.vocab_size
        # Effectively acts as standard multinomial over all tokens

    def test_sample_k_larger_than_vocab_size(self):
        sampler = TopKSampler(k=self.vocab_size + 10) # k > vocab_size
        selected_token = sampler.sample(self.logits)
        assert 0 <= selected_token < self.vocab_size
        # Should behave as if k=vocab_size

    def test_sample_with_filter_tokens_overlap(self):
        # Filter tokens that are a subset of top-k
        k = 10
        filter_tokens = {self.sorted_indices[0].item(), self.sorted_indices[2].item(), self.sorted_indices[5].item()}
        sampler = TopKSampler(k=k)
        
        selected_token = sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert selected_token in filter_tokens
        
        # Verify it picks from the intersection with reasonable probability
        num_trials = 100
        valid_samples = 0
        for _ in range(num_trials):
            sample = sampler.sample(self.logits, filter_tokens=filter_tokens)
            if sample in filter_tokens:
                valid_samples += 1
        assert valid_samples == num_trials # All samples must be from filter_tokens

    def test_sample_with_filter_tokens_no_overlap(self):
        # Filter tokens that are NOT in the top-k initially
        k = 5
        # Ensure filter_tokens are outside the initial top-k
        filter_tokens = {self.sorted_indices[self.vocab_size - 1].item(), self.sorted_indices[self.vocab_size - 2].item()}
        sampler = TopKSampler(k=k)

        selected_token = sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert selected_token in filter_tokens

        # Ensure it falls back to sampling from filter_tokens correctly
        num_trials = 100
        valid_samples = 0
        for _ in range(num_trials):
            sample = sampler.sample(self.logits, filter_tokens=filter_tokens)
            if sample in filter_tokens:
                valid_samples += 1
        assert valid_samples == num_trials

    def test_sample_with_empty_filter_tokens_fallback(self):
        sampler = TopKSampler(k=5)
        # If filter_tokens is empty, the sampler should behave as if no filter was applied.
        # It should sample from the top-k tokens based on the original logits.
        selected_token = sampler.sample(self.logits, filter_tokens=set())
        
        # Get the expected top-k tokens without any filtering
        top_k_expected_tokens = self.sorted_indices[:sampler.k].tolist()
        
        assert selected_token in top_k_expected_tokens

    def test_sample_top_k_with_specific_values(self):
        # Test with known logits and k to ensure correct behavior
        test_logits = torch.tensor([1.0, 5.0, 2.0, 0.5, 4.0, 3.0])
        k = 3 # Top 3 should be 5.0 (idx 1), 4.0 (idx 4), 3.0 (idx 5)
        sampler = TopKSampler(k=k)

        # In multiple trials, samples should only come from indices 1, 4, 5
        allowed_indices = {1, 4, 5}
        num_trials = 100
        for _ in range(num_trials):
            selected_token = sampler.sample(test_logits)
            assert selected_token in allowed_indices

    def test_sample_top_k_with_specific_values_and_filter(self):
        test_logits = torch.tensor([1.0, 5.0, 2.0, 0.5, 4.0, 3.0])
        k = 3 # Top 3: indices {1, 4, 5}
        filter_tokens = {1, 5} # Filter to allow only 1 and 5
        sampler = TopKSampler(k=k)

        # In multiple trials, samples should only come from indices 1, 5 (intersection of top-k and filter)
        allowed_indices = {1, 5}
        num_trials = 100
        for _ in range(num_trials):
            selected_token = sampler.sample(test_logits, filter_tokens=filter_tokens)
            assert selected_token in allowed_indices 