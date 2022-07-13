cmake_minimum_required(VERSION 3.22)
project ( manual NONE)
SET(LATEX_OUTPUT_PATH /home/gordon/openBliSSART/doc/manual_PDF)

INCLUDE(UseLATEX.cmake)

ADD_LATEX_DOCUMENT(manual.tex 
INPUTS overview.tex reference_manual.tex toolbox.tex tutorial.tex 
IMAGE_DIRS images tutorial-media
BIBFILES bibliography.bib IEEEbib.bst)
