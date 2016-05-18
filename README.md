CINI OCR
========

This project aims at digitizing various scanned documents retrieved from the Cini foundation.
The goal is to segment the scans, find the the paintings and text areas inside them, and later be able to extract the text.

Installation
------------

Downloading C libraries: 
 * libzbar-dev, libzbar0
 * Libraw-dev (min version 0.16)
 
Downloading anaconda:
 * wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
 * bash Anaconda3-4.0.0-Linux-x86_64.sh
If you want a different version go to https://www.continuum.io/downloads 
and download from there

Cloning the project and installing the dependencies
 * git clone https://github.com/GrimReaperSam/Cini-OCR.git
 * conda env create -f environment.yml (Make sure there is no opencv3 line in environment.yml, this is because of a bug in anaconda)
 * conda install -n OCR -c menpo opencv3=3.1.0 (OCR is the name of the virtual env)

