
**DOCX Meld Compare**

 - **Problem**
&nbsp;
Sometimes customers provide documentation using .docx files.
It is very uncomfortable to manually find what exactly was changed.
This script can make it easier.
 - **Solution way**
&nbsp;
This script uses Pandoc to transform .docx to .md and Meld to compare changes.
 - **Libs**
&nbsp;
Script will sugest you to install needed libraries automatically, however:
Pandoc is using to transform .docx to .md, so you can to install it manually:
`$ sudo apt install tesseract-ocr`
Meld is using to compare changes:
`$ sudo apt install meld`
 - **Usage**
``compare_docs.sh [-o <old_version.docx>] [-n <new_version.docx>]``
   
Screenshots:
![Example](/screenshots/meld.png?raw=false "Example")
