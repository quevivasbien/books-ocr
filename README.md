# books_ocr
Tools for extracting text from images of books. De-warping is based on code from
mzucker (https://mzucker.github.io/2016/08/15/page-dewarping.html).

To use `ocr.py` to get text from a book, syntax is
`python ocr.py FOLDER OUTPUT_FILE`
where `FOLDER` is the file path to a folder containing `.jpg` images, each image
being a scan of an open book (two pages), and `OUTPUT_FILE` is the name of a
text file to output data to.
