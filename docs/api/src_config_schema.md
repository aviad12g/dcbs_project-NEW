# Configuration Management

Module: `src.config_schema`

## Overview

Configuration schema validation and environment variable support.

This module provides schema validation for YAML configuration files
and support for environment variable substitution.

## Classes

### ConfigSchema

Defines the schema for DCBS configuration files.

#### Constructor

```python
ConfigSchema(args, kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.


### ConfigValidator

Validates configuration against schema.

#### Methods

##### validate

```python
validate(config: Dict)
```

Validate configuration against schema.

Args:
    config: Configuration to validate
    
Returns:
    Validated and normalized configuration
    
Raises:
    ValidationError: If validation fails


### EnvironmentVariableResolver

Handles environment variable substitution in configuration values.

#### Constructor

```python
EnvironmentVariableResolver(args, kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.

#### Methods

##### apply_env_var_overrides

```python
apply_env_var_overrides(config: Dict, schema: Dict)
```

Apply environment variable overrides to configuration.

Args:
    config: Current configuration
    schema: Configuration schema
    
Returns:
    Configuration with environment variable overrides applied

##### resolve_env_vars

```python
resolve_env_vars(value: Any)
```

Resolve environment variables in configuration values.

Args:
    value: Configuration value that may contain environment variables
    
Returns:
    Value with environment variables resolved


## Functions

### generate_config_template

```python
generate_config_template()
```

Generate a template configuration file with documentation.

### validate_config_file

```python
validate_config_file(config_path: str)
```

Load and validate a configuration file.

Args:
    config_path: Path to the configuration file
    
Returns:
    Validated configuration dictionary
    
Raises:
    ConfigurationError: If file cannot be loaded
    ValidationError: If validation fails
