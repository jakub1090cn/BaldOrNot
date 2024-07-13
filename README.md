# BaldOrNot

## Installing Pre-commit Hooks

To ensure code quality and consistency across our project, we use pre-commit hooks that automatically check and format code before commits. Follow the instructions below to set up the pre-commit hooks:

1. **Install Pre-commit**:
   Ensure you have `pre-commit` installed. You can install it via pip:
   ```bash
   pip install pre-commit
   ```

2. **Download the Pre-commit Configuration File**:
   Download the `.pre-commit-config.yaml` file from the root directory of the project repository.

3. **Place the Configuration File**:
   Ensure the downloaded `.pre-commit-config.yaml` file is placed in the root directory of your local project repository.

4. **Install the Pre-commit Hooks**:
   Run the following command in the root directory of the project to install the hooks:
   ```bash
   pre-commit install
   ```

5. **Run Pre-commit Hooks Manually (Optional)**:
   To check all existing files in the repository, you can run the pre-commit hooks manually:
   ```bash
   pre-commit run --all-files
   ```

### Description of Hooks

- **Black**: Automatically formats Python code to adhere to PEP 8 guidelines.
- **Flake8**: Checks for style errors and potential bugs in Python code.
- **Mypy**: Performs static type checking based on type annotations.

By following these steps, you will ensure that your code is automatically formatted and checked for common errors before being committed, helping maintain high code quality throughout the project.
