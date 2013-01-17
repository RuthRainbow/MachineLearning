import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.MetaCost;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;

public class LogisticRegression {

	public static void main(String[] args) {
		
		// Read in the test and training data filepaths from the command line
		int argsLength = args.length;
		if (argsLength < 2) {
			System.out.println("Please specific a training data file and a test data file");
			System.exit(0);
		}
		String testFilename = args[1];
		String trainingFilename = args[0];
		
		Instances trainingInstances = readInstances(trainingFilename);
		Instances testInstances = readInstances(testFilename);

		// Discretise the continuous attributes for both data sets
		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, discreteFilter);
			testInstances = Filter.useFilter(testInstances, discreteFilter);
		} catch (Exception e) {
			System.out.println("Error with Discretize");
			e.printStackTrace();
		}
		
		// Apply SMOTE to the training data for over-sampling
		SMOTE smote = new SMOTE();
		try {
			smote.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, smote);
		} catch (Exception e) {
			System.out.println("Error with SMOTE");
			e.printStackTrace();
		}

		System.out.println("Finished filtering");

		// Create the logistic regression classifier
		Logistic logistic = new Logistic();
		
		// Initialise the cost matrix with the values found
		CostMatrix costMatrix = new CostMatrix(2);
		costMatrix.setElement(0, 0, 0);
		costMatrix.setElement(0, 1, 1);
		costMatrix.setElement(1, 0, 5);
		costMatrix.setElement(1, 1, 0);
		
		// Build the MetaCost classifier using the Logistic classifier and the cost matrix defined above
		MetaCost metaCost = new MetaCost();
		metaCost.setClassifier(logistic);
		metaCost.setCostMatrix(costMatrix);
		try {
			metaCost.buildClassifier(trainingInstances);
		} catch (Exception e1) {
			System.out.println("Error building the MetaCost classifier");
			e1.printStackTrace();
		}
		
		// Evaluate the model using the test data and print out the results
		Evaluation evaluation = null;
		try {
			evaluation = new Evaluation(trainingInstances);
			evaluation.evaluateModel(metaCost, testInstances);
			System.out.println(evaluation.toSummaryString());
			System.out.println(evaluation.toClassDetailsString());
			System.out.println(evaluation.toMatrixString());
		} catch (Exception e) {
			System.out.println("Error outputting evaluation");
		}
	}
	
	// Read instances from the given file name and return these
	private static Instances readInstances(String file) {
		DataSource source = null;
		try {
			source = new DataSource(file);
		} catch (Exception e) {
			System.out.println("could not find file");
			e.printStackTrace();
		}

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
