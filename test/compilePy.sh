pyinstaller --onefile numberreader.py
pyinstaller --onefile --add-data "boundingbox.py:." --add-data "neuralnetwork.py:." numberreader.py
