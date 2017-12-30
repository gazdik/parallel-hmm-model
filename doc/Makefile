NAME=doc

$(NAME).pdf: $(NAME).tex
	pdflatex $(NAME)
	bibtex $(NAME)
	pdflatex $(NAME)
	pdflatex $(NAME)

clean:
	-rm -f *.dvi *.aux *.toc *.log *.out *.pdf *.bbl *.blg images/*converted*pdf
