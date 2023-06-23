# Neuronales Netzwerk

Für die Begabtenförderung habe ich ein neuronales Netzwerk programmiert und trainiert, welches handgeschriebene Ziffern erkennen soll.
Dabei habe ich das Buch https://neuralnetworksanddeeplearning.com/ verwended, um ein Verständnis für neuronale Netzwerke zu bekommen. 
Die Daten habe ich von http://yann.lecun.com/exdb/mnist/ bezogen und einige auch selbst hinzugefügt.

## Ausprobieren
Das Netzwerk kann in einem Terminal durch ```./dist/numberreader/numberreader``` ausprobiert werden. Es kann eine Weile dauern, bis das Programm startet, aber dann sollte sich ein Fenster öffnen.
Dort kann in das schwarze Feld eine Ziffer geschrieben werden, welche dann durch das Network mit ```Enter``` predicted wird. Diese Zahl kann auch ins Trainingsset gegeben werden, indem man ```s``` drückt und dann die richtige Zahl eingibt (es gibt keine Möglichkeit diese Eingabe zu löschen also sollte darauf geachtet werden, dass man die korrekte Zahl angibt). Mit ```c``` kann das Fenster gecleared werden und eine neue Ziffer gezeichnet werden.

## Selbst Trainieren
Trainieren kann man es ebenfalls in einem Terminal durch ```network.txt```. Hier können als freiwillige Argumente die learning rates (fully connected weights, fully connected biases, convolutional weights, convolutional biases, L2 regularization und momentum coefficient) gegeben werden. Allerding müssen entweder alle (in der beschriebenen Reihenfolge) oder keine gegeben werden.
Alle Zahlen sollten positiv sein und L2 regularization sowie momentum coefficient sollten kleiner als 1 sein. Auch wenn diese Begriffe nicht bekannt sind, kann gerne ausprobiert werden wie diese Werte das Programm beeinflussen (ziemlich fest).

Ausserdem wird das Programm beim ausführen den User für ```epochs``` prompten. Dieser Wert gibt an, wie lange das Netzwerk trainieren wird und sollte deshalb ein Integer > 0 sein.

Das Programm wird einige Informationen als Output geben. Nach jedem Durchlauf (Epoch) wird die Genauigkeit (im Testset) ausgegeben. Am Ende wird die Genauigkeit im Trainingsset und im Testset ausgegeben.

Dieses Programm wird auch eine Weile dauern, da es viele Daten verarbeiten muss.



