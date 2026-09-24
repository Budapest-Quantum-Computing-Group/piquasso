#!/bin/bash

cd ./tutorials/figures

pdflatex one-mode-gate-teleport-light.tex 
pdflatex one-mode-gate-teleport-dark.tex

pdflatex two-mode-gate-teleport-light.tex 
pdflatex two-mode-gate-teleport-dark.tex 


pdf2svg one-mode-gate-teleport-light.pdf one-mode-gate-teleport-light.svg
pdf2svg one-mode-gate-teleport-dark.pdf one-mode-gate-teleport-dark.svg


pdf2svg two-mode-gate-teleport-light.pdf two-mode-gate-teleport-light.svg
pdf2svg two-mode-gate-teleport-dark.pdf two-mode-gate-teleport-dark.svg


rm *.aux
rm *.log
if [ -z $1 ]
then
    rm *.pdf
fi

echo $1