# Contributing to VisuaML

Thank you for your interest in contributing to VisuaML! This guide will help you get started with contributing to our interactive neural network visualization platform.

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and **pnpm** (repo is a pnpm workspace)
- **Python** 3.11+ (only if you need the API/model import/export; see [visuaml-client/PYTHON_SETUP.md](../visuaml-client/PYTHON_SETUP.md))
- **Git** for version control
- **Basic knowledge** of React, TypeScript, and PyTorch

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/Caerii/VisuaML.git
   cd VisuaML
   ```

2. **Install dependencies** (from repo root)
   ```bash
   pnpm install
   ```
   Optional, for API and model import/export:
   ```bash
   pnpm run install-python-deps
   ```

3. **Set up environment**
   ```bash
   cd visuaml-client
   cp env.example .env.local
   # Edit .env.local: VISUAML_PYTHON, Clerk publishable key, etc.
   ```

4. **Start development servers** (from repo root, one per terminal)
   ```bash
   pnpm --filter visuaml-client run api    # Terminal 1: API (needs Python deps)
   pnpm --filter visuaml-client run ws     # Terminal 2: WebSocket (multiplayer)
   pnpm --filter visuaml-client run dev   # Terminal 3: Frontend
   ```

## 📋 How to Contribute

### Types of Contributions

- **🐛 Bug Reports**: Found a bug? Please report it!
- **✨ Feature Requests**: Have an idea? We'd love to hear it!
- **📝 Documentation**: Help improve our docs
- **🔧 Code Contributions**: Fix bugs or implement features
- **🧪 Testing**: Add tests or improve test coverage
- **🎨 UI/UX**: Improve the user interface and experience

### Before You Start

1. **Check existing issues** to avoid duplicates
2. **Create an issue** for new features or major changes
3. **Discuss your approach** in the issue before coding
4. **Keep changes focused** - one feature/fix per PR

## 🛠️ Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

Follow our coding standards (see below) and make your changes.

### 3. Test Your Changes

From repo root (client tests):
```bash
pnpm --filter visuaml-client run test
pnpm --filter visuaml-client run lint
# type-check if available: pnpm --filter visuaml-client run type-check
```

Backend tests (from visuaml-client directory):
```bash
cd visuaml-client/backend && python -m pytest
# Integration: from visuaml-client, python test_export_frontend.py (if present)
```

### 4. Commit Your Changes

We use conventional commits:

```bash
git commit -m "feat: add new export format"
git commit -m "fix: resolve shape propagation issue"
git commit -m "docs: update API documentation"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 📏 Coding Standards

### TypeScript/React

- **Use TypeScript strictly** - no `any` types
- **Functional components** with hooks
- **File naming**: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- **Export patterns**: Named exports preferred, default for main component
- **Documentation**: JSDoc comments for public functions

```typescript
/** @fileoverview Brief description of the file's purpose. */

/**
 * Processes export results into archive-ready files.
 * 
 * @param allResults - The export results from the API
 * @returns Array of archive files ready for packaging
 */
export const processExportResults = (
  allResults: AllExportsData
): ArchiveFile[] => {
  // Implementation
};
```

### Python

- **Follow PEP 8** style guidelines
- **Type hints** for all function parameters and returns
- **Docstrings** for all modules, classes, and functions
- **Error handling** with specific exception types

```python
"""Module for exporting PyTorch models to VisuaML graph format."""

from typing import Dict, List, Optional

def export_model_graph(
    model_path: str,
    filter_config: Optional[FilterConfig] = None,
) -> Dict[str, Any]:
    """
    Export a PyTorch model to VisuaML graph format.
    
    Args:
        model_path: Module path to the model (e.g., 'models.MyModel')
        filter_config: Optional filter configuration
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
        
    Raises:
        ModelLoadError: If model cannot be loaded
    """
    # Implementation
```

### File Organization

```
src/
├── ui/                     # UI components
│   ├── ComponentName/      # Component directory
│   │   ├── ComponentName.tsx
│   │   ├── ComponentName.model.ts
│   │   ├── useComponentName.ts
│   │   └── index.ts
├── lib/                    # Utilities
│   ├── utilityName.ts
│   └── README.md
└── store/                  # State management
```

## 🧪 Testing Guidelines

### Frontend Testing

- **Unit tests** for utility functions
- **Component tests** for UI components
- **Integration tests** for workflows
- **Type tests** for TypeScript interfaces

```typescript
// Example test
describe('processExportResults', () => {
  it('should convert API results to archive files', () => {
    const mockResults = createMockExportResults();
    const files = processExportResults(mockResults);
    
    expect(files).toHaveLength(4);
    expect(files[0].name).toMatch(/\.json$/);
  });
});
```

### Backend Testing

- **Unit tests** for individual functions
- **Integration tests** for model export workflows
- **Mock tests** for external dependencies

```python
def test_export_model_graph():
    """Test basic model export functionality."""
    result = export_model_graph('models.SimpleNN')
    
    assert 'nodes' in result
    assert 'edges' in result
    assert len(result['nodes']) > 0
```

## 📝 Documentation

### Code Documentation

- **File overviews** for all TypeScript files
- **Function documentation** with JSDoc/docstrings
- **Type definitions** with clear interfaces
- **Usage examples** in README files

### API Documentation

When adding new API endpoints:

1. **Document the endpoint** in the main README
2. **Add request/response examples**
3. **Update the OpenAPI spec** (if we add one)
4. **Test the endpoint** with integration tests

## 🎯 Specific Contribution Areas

### Adding New Export Formats

1. **Backend**: Add export logic in `backend/visuaml/`
2. **API**: Add endpoint in `server/index.ts`
3. **Frontend**: Add UI controls in `TopBar`
4. **Types**: Update TypeScript interfaces
5. **Tests**: Add comprehensive tests
6. **Docs**: Update documentation

### Adding New Model Types

1. **Create model** in `backend/models/`
2. **Add to model list** in `TopBar.model.ts`
3. **Configure sample inputs** in `server/index.ts`
4. **Test compatibility** with all export formats
5. **Document usage** in README

### UI/UX Improvements

1. **Follow Material-UI** design patterns
2. **Ensure accessibility** (ARIA labels, keyboard navigation)
3. **Test responsiveness** on different screen sizes
4. **Maintain consistency** with existing design
5. **Add proper loading states** and error handling

## 🔍 Code Review Process

### What We Look For

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it properly documented?
- **Performance**: Does it impact performance?
- **Security**: Are there any security concerns?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is acceptable
- [ ] Security considerations are addressed

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for duplicates
2. **Try the latest version** to see if it's fixed
3. **Gather information** about your environment

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Windows 10]
- Browser: [e.g. Chrome 91]
- Node.js version: [e.g. 18.0.0]
- Python version: [e.g. 3.9.0]

**Additional context**
Any other context about the problem.
```

## ✨ Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Tag maintainers for review help

## 🏆 Recognition

Contributors will be:
- **Listed in CONTRIBUTORS.md**
- **Given credit in documentation**
- **Mentioned in release notes**

Thank you for contributing to VisuaML! 🎉 