import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.MetaCost;
import weka.classifiers.rules.Prism;

public class RuleInduction {

	public static void main(String[] args) {
		
		String train = args[0];
		String test = args[1];
		
		DataSource testData = null;
		DataSource source = null;
		
		// Set the data sources to the user supplied training and test data
		try {
			source = new DataSource(train);
			testData = new DataSource(test);
		} catch (Exception e) {
			System.out.println("could not find file");
			e.printStackTrace();
		}

		// Get the instances from the supplied data sources
		Instances instances = null;
		Instances testInstances = null;
		
		try {
			instances = source.getDataSet();
			testInstances = testData.getDataSet();
		} catch (Exception e) {
			System.out.println("error getting data set from source");
			e.printStackTrace();
		}

		// Set the class index as the "SeriousDlqin2yrs" attribute
		instances.setClassIndex(0);
		testInstances.setClassIndex(0);
		
		// Replace the missing values in the training and test data with means
		ReplaceMissingValues replaceFilter = new ReplaceMissingValues();
		try {
			replaceFilter.setInputFormat(instances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
			instances = Filter.useFilter(instances, replaceFilter);
			testInstances = Filter.useFilter(testInstances, replaceFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Discretise the continuous attributes in the training and test data
		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(instances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
			instances = Filter.useFilter(instances, discreteFilter);
			testInstances = Filter.useFilter(testInstances, discreteFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// Use SMOTE to resample the training data
		SMOTE smote = new SMOTE();
		try {
			smote.setInputFormat(instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		try {
			instances = Filter.useFilter(instances, smote);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// Set up the Prism classifier with a cost matrix
		Prism prism = new Prism();
		CostMatrix costMatrix = new CostMatrix(2);
		
		costMatrix.setElement(0, 0, 0);
		costMatrix.setElement(0, 1, 1);
		costMatrix.setElement(1, 0, 10);
		costMatrix.setElement(1, 1, 0);

		MetaCost metaCost = new MetaCost();
		metaCost.setClassifier(prism);
		metaCost.setCostMatrix(costMatrix);
		
		try {
			metaCost.buildClassifier(instances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		// Evaluate the test data, printing information about the results
		Evaluation eval;
		try {
			eval = new Evaluation(instances);
			eval.evaluateModel(metaCost, testInstances);
			System.out.println(eval.toSummaryString("\nResults\n", false));
			System.out.println(eval.toMatrixString());
			System.out.println(eval.toClassDetailsString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}
}
