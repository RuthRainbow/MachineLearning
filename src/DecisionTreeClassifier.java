import java.io.File;
import java.util.Random;

import weka.classifiers.trees.J48;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.Evaluation;

public class DecisionTreeClassifier {
	public static void main(String[] args) {
		Instances creditData = null;
		Instances testData = null;
		try {
			ArffLoader loader = new ArffLoader();
			loader.setSource(new File("Data.arff"));
			creditData = loader.getDataSet();
			ArffLoader testLoader = new ArffLoader();
			testLoader.setSource(new File("testdata.arff"));
			testData = testLoader.getDataSet();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		creditData.setClassIndex(0);
		testData.setClassIndex(0);
		
		Discretize discretize = new Discretize(); // new instance of filter
		try {
			discretize.setInputFormat(creditData); // inform filter about dataset
			// **AFTER** setting options
			creditData = Filter.useFilter(creditData, discretize); // apply filter
			testData= Filter.useFilter(testData, discretize); // apply filter
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} // set options
		
		Resample resample = new Resample(); // new instance of filter
		try {
			resample.setInputFormat(creditData); // inform filter about dataset
			resample.setBiasToUniformClass(1.0); 
			// **AFTER** setting options
			creditData = Filter.useFilter(creditData, resample); // apply filter
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} // set options
		
		String[] options = new String[1];
	    options[0] = "-R"; 
		J48 tree = new J48(); // new instance of tree
		try {
			tree.setOptions(options); // set the options
			tree.buildClassifier(creditData); // build classifier
		}
		catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// evaluate classifier and print some statistics
		Evaluation eval;
		try {
			eval = new Evaluation(creditData);
			eval.evaluateModel(tree, testData);
			System.out.println(eval.toSummaryString("\nResults\n\n", false));
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}	
}
