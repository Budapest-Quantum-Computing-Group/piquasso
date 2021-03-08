# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS		?=
SPHINXBUILD		:= sphinx-build
SOURCEDIR		:= .
BUILDDIR		:= _build

#variables for sphinx-apidoc
SPHINX_APIDOC	:= sphinx-apidoc
APIDOC_EXCLUDE_PATTERN := ../piquasso/tests/*
APIDOC_MODULE_PATH		:= ../piquasso
APIDOC_OUTPUT_PATH		:= .


.PHONY: help apidoc html html_all open Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "------------------------------------------------------------------------------------------------------------------------"
	@echo "run <poetry run make html_all> in the sphinx root directory (/docs folder) to generate the web page from the docsstrings"

apidoc:
	@$(SPHINX_APIDOC) -f -o "$(APIDOC_OUTPUT_PATH)" "$(APIDOC_MODULE_PATH)" "$(APIDOC_EXCLUDE_PATTERN)"

clean:
	rm -rf "$(BUILDDIR)"

html: clean
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)"

html_all: apidoc html

open: html
	@xdg-open $(BUILDDIR)/index.html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
