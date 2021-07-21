**_Google Meet Flood Finder_**\
<br/>
Sometimes it is pretty hard to detect who is talking too much on meetings.
<br/>

This application uses MSS to take a screenshot of Google Meet every second,
OpenCV to find an active speaker and Tesseract to OCR his name.
<br/>
During meeting and after meeting close it will show you a detailed statistic.
<br/>

###### **Libs**
App uses OCR to get information about current speaker, so you need to install it:
<br/>
`# $ sudo apt install tesseract-ocr`
<br/>

All Python libs can be installed by executing:
<br/>
`pip install -r requirements.txt`