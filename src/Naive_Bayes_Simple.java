
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.CostMatrix;
import weka.classifiers.meta.MetaCost;
import weka.classifiers.Evaluation;


public class Naive_Bayes_Simple {
	public static void main(String[] args) {
		
		// Analyse arguments for input data
		int argsLength = args.length;
		if (argsLength < 2) {
			System.out.println("Please specific a training data file and a test data file");
			System.exit(0);
		}
		String testFilename = args[1];
		String trainingFilename = args[0];
		
		// Gather the instances from the data files
		Instances data = readInstances(trainingFilename);
		Instances testData = readInstances(testFilename);
		
		// Discretise the continuous attributes for both data sets
		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(data);
			data = Filter.useFilter(data, discreteFilter);
			testData = Filter.useFilter(testData, discreteFilter);
		} catch (Exception e) {
			System.out.println("Error with Discretize");
			e.printStackTrace();
		}
		
		// Create a SMOTE instance
		SMOTE smoter = new SMOTE();
		try {
			smoter.setNearestNeighbors(2);
			smoter.setPercentage(100);
		} catch (Exception e) {
			System.out.println("Cannot create smote");
		}

		// Run the training data through the SMOTE filter
		try {
			smoter.setInputFormat(data);
			data = Filter.useFilter(data, smoter);			
		} catch (Exception e) {
			System.out.println("Cannot apply smote");
		}	
		
		// Build a naive bayes classifier for the data
		NaiveBayesSimple simpleNaiveBayes = new NaiveBayesSimple();
		try {
			simpleNaiveBayes.buildClassifier(data);
		} catch (Exception e) {
			System.out.println("Cannot build classifer");
		}
		
		// Build cost matrix
		CostMatrix costMatrix = new CostMatrix(2);
		costMatrix.setElement(0, 0, 0);
		costMatrix.setElement(0, 1, 1);
		costMatrix.setElement(1, 0, 11);
		costMatrix.setElement(1, 1, 0);

		// Instantiate the MetaCost classifier and set the base classifier as the Simple Naive Bayes classifer
		MetaCost metaCost = new MetaCost();
		metaCost.setClassifier(simpleNaiveBayes);
		metaCost.setCostMatrix(costMatrix);
		
		// Build the MetaCost classifier
		try {
			metaCost.buildClassifier(data);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		// evaluate classifier and print some statistics
		Evaluation eval;
		try {
			eval = new Evaluation(data);
			eval.evaluateModel(simpleNaiveBayes, testData);
			System.out.println(eval.toSummaryString("\nResults\n\n", false));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			System.out.println("Cannot evaluate classifier");
		}

	}
	
	private static Instances readInstances(String file) {
		// Instantiate a datasource
		DataSource source = null;
		try {
			source = new DataSource(file + ".arff");
		} catch (Exception e) {
			System.out.println("could not find file");
			e.printStackTrace();
		}

		// Gather the instances
		Instances instances = null;
		try {
			instances = source.getDataSet();
		} catch (Exception e) {
			System.out.println("error getting data set from source");
			e.printStackTrace();
		}

		// Set the class index as the "SeriousDlqin2yrs" attribute
		instances.setClassIndex(0);

		return instances;
	}
}
