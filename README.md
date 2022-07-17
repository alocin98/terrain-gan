# Terrain Gan

In dieser Arbeit wird ein GAN vorgeschlagen, welches natürliche Landschaften generieren kann. Als Inputdaten werden Höhenmodelle des gesamten Alpenraums genommen. Bei der Suche nach dem besten Modell werden drei verschiedene GAN Architekturen mit verschiedenen Parameter miteinander verglichen und getestet. Als Parameter werden Lossfunktion, Optimierungsverfahren und Inputdaten variiert. Während des Trainings werden Modelle und Parameter, welche schlechte Ergebnisse liefern, verworfen. Dazu gehören ein GAN ohne Convolutional Layer, das Stochastic Gradient Descent Optimierungsverfahren und die 'Mean Squared Error' Lossfunktion. Die restlichen Parameter liefern vergleichbare Ergebnisse, wobei das finale Modell dieser Arbeit ein Deep Convolutional Generative Adversarial Network ist, welches mit der Binarycrossentropy Lossfunktion und dem Adam Optimierungsverfahren trainiert wurde. Es folgt eine 3-Dimensionale Visualisierung eines generierten Höhenmodells sowie ein Ausblick über die mögliche weiterführende Forschung.

## Model
Das finale Modell ist unter models/ zu finden. Es wurde im .h5 Format abgespeichert

## Daten
Die Daten sind unter data/ zu finden

## Notebooks
Notebooks, welche zum Beispiel zum Trainineren benutzt wurden, sind unter notebooks/ zu finden.