# Requirements Document

## Introduction

This specification addresses critical issues in the DCBS evaluation system that are preventing successful evaluations and causing inefficiencies. The system currently fails with JSON serialization errors, lacks proper parameter tracking, has incorrect chain-of-thought parsing, and performs redundant evaluations.

## Requirements

### Requirement 1: JSON Serialization Fix

**User Story:** As a researcher, I want the evaluation system to save checkpoints and results without JSON serialization errors, so that my long-running evaluations can be resumed and results can be properly stored.

#### Acceptance Criteria

1. WHEN the system attempts to save checkpoint data THEN it SHALL convert all numpy int64 and other non-JSON-serializable types to native Python types
2. WHEN the system encounters numpy arrays THEN it SHALL convert them to Python lists before JSON serialization
3. WHEN the system encounters numpy scalars THEN it SHALL convert them to native Python int/float types
4. IF any object contains nested non-serializable types THEN the system SHALL recursively convert all nested objects
5. WHEN checkpoint saving fails THEN the system SHALL log the specific object type causing the failure and attempt graceful degradation

### Requirement 2: Parameter Tracking Enhancement

**User Story:** As a researcher, I want all sampling algorithm parameters to be recorded in evaluation results, so that I can reproduce experiments and understand the configuration used for each result.

#### Acceptance Criteria

1. WHEN an evaluation is performed THEN the system SHALL record all DCBS parameters including k, top_n, clustering_method, eps, min_samples
2. WHEN multiple clustering methods are evaluated THEN each result SHALL include the specific clustering parameters used
3. WHEN greedy sampling is performed THEN the system SHALL record any relevant parameters (temperature, top_p values if applicable)
4. WHEN results are saved THEN the parameter information SHALL be included in both detailed results and summary statistics
5. WHEN loading results for analysis THEN all original parameters SHALL be available for comparison and reproduction

### Requirement 3: Chain-of-Thought Parsing Fix

**User Story:** As a researcher, I want the chain-of-thought reasoning to be correctly extracted from model responses, so that I can analyze the quality of reasoning and debug model behavior.

#### Acceptance Criteria

1. WHEN the model generates a reasoning response THEN the system SHALL extract only the actual reasoning content, not template text
2. WHEN the response contains template phrases like "LM that thinks step by step before answering" THEN the system SHALL exclude these from the reasoning extraction
3. WHEN parsing CoT responses THEN the system SHALL identify and extract the substantive reasoning portion
4. IF the CoT response is malformed or empty THEN the system SHALL handle this gracefully and log appropriate warnings
5. WHEN storing CoT results THEN the system SHALL validate that meaningful reasoning content was extracted

### Requirement 4: Evaluation Efficiency Optimization

**User Story:** As a researcher, I want to avoid redundant evaluations when testing multiple clustering methods, so that my experiments run faster and don't waste computational resources.

#### Acceptance Criteria

1. WHEN multiple clustering methods are evaluated with the same baseline samplers THEN the system SHALL run baseline evaluations only once
2. WHEN greedy sampling results are available from a previous clustering method evaluation THEN the system SHALL reuse those results
3. WHEN caching baseline results THEN the system SHALL ensure cache keys include all relevant parameters to avoid incorrect reuse
4. WHEN running multi-dataset evaluations THEN the system SHALL optimize the evaluation order to maximize result reuse
5. WHEN baseline results are reused THEN the system SHALL log this optimization and verify parameter compatibility

### Requirement 5: Error Handling and Recovery

**User Story:** As a researcher, I want the evaluation system to handle errors gracefully and provide clear diagnostic information, so that I can identify and fix issues without losing evaluation progress.

#### Acceptance Criteria

1. WHEN JSON serialization fails THEN the system SHALL provide detailed error messages indicating the problematic object type and location
2. WHEN an individual dataset evaluation fails THEN the system SHALL continue with remaining datasets and report partial results
3. WHEN checkpoint loading fails THEN the system SHALL attempt to recover partial state and continue from the last valid checkpoint
4. WHEN parameter validation fails THEN the system SHALL provide clear error messages indicating which parameters are invalid
5. WHEN evaluation errors occur THEN the system SHALL save diagnostic information to help with debugging

### Requirement 6: Result File Structure Enhancement

**User Story:** As a researcher, I want evaluation result files to have a consistent, comprehensive structure that includes all necessary metadata, so that I can easily analyze and compare results across different experiments.

#### Acceptance Criteria

1. WHEN saving evaluation results THEN the system SHALL include a standardized metadata section with timestamp, git commit, system info
2. WHEN multiple samplers are evaluated THEN each sampler's results SHALL include its complete parameter configuration
3. WHEN clustering methods are compared THEN the results SHALL clearly indicate which clustering method was used for each result
4. WHEN statistical analysis is performed THEN confidence intervals and significance tests SHALL be included in the results
5. WHEN results are aggregated THEN the system SHALL maintain traceability to individual evaluation runs

### Requirement 7: Configuration Validation

**User Story:** As a researcher, I want the system to validate evaluation configurations before starting long-running experiments, so that I can catch configuration errors early and avoid wasted computation.

#### Acceptance Criteria

1. WHEN an evaluation is started THEN the system SHALL validate all sampler parameters are within acceptable ranges
2. WHEN clustering parameters are specified THEN the system SHALL verify they are compatible with the selected clustering method
3. WHEN dataset parameters are provided THEN the system SHALL verify the datasets exist and are accessible
4. WHEN batch size is specified THEN the system SHALL validate it's appropriate for available GPU memory
5. IF configuration validation fails THEN the system SHALL provide specific error messages and suggested corrections