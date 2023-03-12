#!/bin/sh

pandoc This\ Is\ Not\ Machine\ Learning/Introduction.tex -f latex -t rst -s -o course/source/Introduction/Introduction.rst

pandoc This\ Is\ Not\ Machine\ Learning/Complexity.tex -f latex -t rst -s -o course/source/Complexity/Complexity.rst
sed -i -E 's/:raw-latex:`\\cite{\(.*\)}`/ :cite:p:`\1`/' course/source/Complexity/Complexity.rst 
sed -i -e '$s/$/\n\n.. bibliography:: ../refs.bib/' course/source/Complexity/Complexity.rst

pandoc This\ Is\ Not\ Machine\ Learning/Closed.tex -f latex -t rst -s -o course/source/Closed/Closed.rst
pandoc This\ Is\ Not\ Machine\ Learning/Open.tex -f latex -t rst -s -o course/source/Open/Open.rst
pandoc This\ Is\ Not\ Machine\ Learning/Relational.tex -f latex -t rst -s -o course/source/Relational/Relational.rst


cp -r This\ Is\ Not\ Machine\ Learning/refs.bib course/source/
cp -r This\ Is\ Not\ Machine\ Learning/Figures course/source/Complexity/
cp -r This\ Is\ Not\ Machine\ Learning/Figures course/source/Closed/