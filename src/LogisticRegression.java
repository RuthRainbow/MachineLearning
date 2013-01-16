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
		
		int argsLength = args.length;
		if (argsLength < 2) {
			System.out.println("Please specific a training data file and a test data file");
			System.exit(0);
		}
		String testFilename = args[1];
		String trainingFilename = args[0];
		

		Instances trainingInstances = readInstances(trainingFilename);
		Instances testInstances = readInstances(testFilename);

		// Discretise the continuous attributes
		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, discreteFilter);
			testInstances = Filter.useFilter(testInstances, discreteFilter);
		} catch (Exception e) {
			System.out.println("Error with Discretize");
			e.printStackTrace();
		}
		
		SMOTE smote = new SMOTE();
		try {
			smote.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, smote);
		} catch (Exception e) {
			System.out.println("Error with SMOTE");
			e.printStackTrace();
		}

		System.out.println("finished filtering");

		Logistic logistic = new Logistic();
		
		CostMatrix costMatrix = new CostMatrix(2);
		costMatrix.setElement(0, 0, 0);
		costMatrix.setElement(0, 1, 1);
		costMatrix.setElement(1, 0, 5);
		costMatrix.setElement(1, 1, 0);
		
		MetaCost metaCost = new MetaCost();
		metaCost.setClassifier(logistic);
		metaCost.setCostMatrix(costMatrix);
		try {
			metaCost.buildClassifier(trainingInstances);
		} catch (Exception e1) {
			System.out.println("Error building the MetaCost classifier");
			e1.printStackTrace();
		}

		System.out.println("built the classifier");
		
		// Print out the results
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
	
	private static Instances readInstances(String file) {
		DataSource source = null;
		try {
			source = new DataSource(file + ".arff");
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
