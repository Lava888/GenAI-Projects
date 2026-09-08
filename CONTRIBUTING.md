# Contributing to GenAI Projects

Thank you for your interest in contributing! Here are guidelines to help you get started.

## 🐛 Reporting Issues

- Check existing issues before creating a new one
- Provide clear description of the problem
- Include:
  - Python version and dependencies
  - Error messages and stack traces
  - Steps to reproduce
  - Your environment (OS, GPU/CPU, etc.)

## 🚀 Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Keep commits atomic and well-described
   - Follow Python best practices (PEP 8)
   - Add comments for complex logic
4. **Test your changes**
   - Run notebooks/scripts locally
   - Verify no API keys are exposed
5. **Commit with clear messages**
   ```bash
   git commit -m "Add: description of your changes"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**
   - Provide detailed description
   - Link related issues
   - Describe testing performed

## 📋 Code Style

### Python
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where applicable
- Max line length: 100 characters

### Jupyter Notebooks
- Clear cell structure
- Include markdown explanations
- No cell outputs in commits (use `.gitignore`)
- Document dependencies at top

### Comments
```python
# Good: Explain WHY, not WHAT
# Use exponential backoff to avoid rate limiting
time.sleep(2 ** attempt)

# Bad: Obvious from code
# Increment counter
counter += 1
```

## 📚 Documentation

- Update README if adding new projects
- Include docstrings in code
- Add usage examples
- Document prerequisites and setup
- Include troubleshooting section
- Update requirements.txt for new dependencies

## ✅ Checklist Before Submitting

- [ ] Code follows style guidelines
- [ ] Self-reviewed your changes
- [ ] Added comments for complex logic
- [ ] Updated documentation and README
- [ ] No new warnings generated
- [ ] No API keys or secrets in code
- [ ] Added dependencies to requirements.txt
- [ ] Tested locally
- [ ] Related issues linked

## 🎯 Types of Contributions

We welcome:
- ✅ New AI/ML projects and agents
- ✅ Improvements to existing code
- ✅ Better documentation
- ✅ Bug fixes
- ✅ Performance optimizations
- ✅ Example notebooks
- ✅ Community feedback

## 📁 Adding New Projects

### Step 1: Create Project Directory
```bash
mkdir project-name
cd project-name
echo "# Project Name Description" > README.md
```

### Step 2: Add Project Files
- Python scripts: `scripts/`, `src/`, or root
- Notebooks: `notebooks/`
- Data: `data/` (add to .gitignore)
- README.md with setup instructions

### Step 3: Update Main README
1. Add project to Projects Overview table
2. Include description and tech stack
3. Link to project README

### Step 4: Test & Submit
```bash
# Test your code
python your_script.py

# Commit and push
git add .
git commit -m "Add: Your project name and description"
git push origin feature/new-project
```

## 🔍 Code Review Process

1. Maintainers will review your PR
2. Feedback and suggestions may be provided
3. Make requested changes
4. Once approved, PR will be merged

## ❓ Questions?

Open a GitHub issue or discussion if you need clarification on:
- How to get started
- Design decisions
- Architecture questions
- Feature requests

---

Thank you for contributing! 🎉
