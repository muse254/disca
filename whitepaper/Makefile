# Makefile for LaTeX white paper

TARGET = main

build:
	xelatex $(TARGET).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz *.pdf

cleanall: clean
	rm -f $(TARGET).pdf