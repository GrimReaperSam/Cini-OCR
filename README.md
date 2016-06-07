CINI OCR
========

This project aims at digitizing various scanned documents retrieved from the Cini foundation.
The goal is to segment the scans, find the the paintings and text areas inside them, and later be able to extract the text.

Installation
------------

Downloading C libraries: 
 * libzbar-dev, libzbar0
 
Downloading anaconda:
 * wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
 * bash Anaconda3-4.0.0-Linux-x86_64.sh
If you want a different version go to https://www.continuum.io/downloads 
and download from there

Cloning the project and installing the dependencies
 * git clone https://github.com/GrimReaperSam/Cini-OCR.git
 * conda env create -f environment.yml
 * conda install -n OCR -c menpo opencv3=3.1.0 (OCR is the name of the virtual env)
 * conda install -n OCR -c atanahel rawkit=0.5.0

Running
-------
 
To run you can use the following command:
 * python pipeliner.py
The arguments are:
 * '-r' Raws directory
 * '-d' Destination directory
 * '-s' Skip processed

Project Structure
-----------------
 
The main files in the project are the following:
 * shared.py: Contains some shared constants
 * raw_converter.py: Converts a RAW CR2 file into a numpy array
 * document.py: Detects the cardboard inside the image and crops it out
 * cardboard.py: Detects the painting and the text section inside the cardboard and crops them out
 * barcode.py: Detects the barcode area in a verso and reads it
 * extractor.py: Given a text section finds the different boxes inside it, and the location of the text.
                 Then using an OCR, it reads it and creates a bounds+text structure
 * pipeline.py: Groups all the previous classes in a pipeline. Takes a folder of raws images and processes them one by one.
                Saves each recto/verso pair into a folder with all their extracted information.
 

