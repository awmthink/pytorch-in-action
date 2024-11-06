# Makefile for generating Sphinx documentation

# Variables
SPHINXOPTS    = -T -b html -d build/doctrees -D language=zh-cn
SPHINXBUILD   = python -m sphinx
SOURCEDIR     = .
BUILDDIR      = build/html

# Targets
.PHONY: html clean

# Generate HTML documentation
html:
	$(SPHINXBUILD) $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)

# Clean the build directory
clean:
	rm -rf $(BUILDDIR) build/doctrees
