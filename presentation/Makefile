NAME=presentation

$(NAME).pdf: $(NAME).tex
	xelatex -shell-escape $(NAME)
	# bibtex $(NAME)
	# pdflatex $(NAME)
	xelatex -shell-escape $(NAME)

clean:
	-rm -f *.dvi *.aux *.toc *.log *.out *.pdf *.nav *.snm *.gnuplot *.table *.bbl img/*.pdf fig/*.pdf *.blg *.zip

pack:
	zip -r gmu-presentation.zip *.tex *.sty fig/ img/ czechiso.bst package/ Makefile presentation.pdf
