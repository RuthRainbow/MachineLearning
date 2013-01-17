How to run our models
The models must be compiled with the Weka jar as a dependency. The Weka version used is 3.6.8 (the version that contains Prism and SimpleNaiveBayes). From the source code directory run the command:

javac -cp [path to weka.jar] [model.java]
java -cp [path to weka.jar]: [model] [path to training data file] [path to test data file]

for example, java -cp ~/weka.jar: LogisticRegression ../trainingdata ../testdata

The test and training data files must be in ARFF format.
