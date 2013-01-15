import java.util.Random;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.classifiers.Evaluation;
import weka.filters.supervised.instance.Resample;

public class Naive_Bayes_Simple {
	public static void main(String[] args) {
		DataSource source = null;
		try {
			source = new DataSource("./credit1.arff");
		} catch (Exception e) {
			System.out.println("Could not load datasource");
		}
		
		Instances data = null;
		try {
			data = source.getDataSet();
		} catch (Exception e) {
			System.out.println("Instances could not be loaded.");
		}
		
		// Set the class index as the "SeriousDlqin2yrs" attribute
		data.setClassIndex(0);
		
		String[] options = new String[2];
		options[0] = "-B";
		options[1] ="1";
		Resample resampler = new Resample();
		try {
			resampler.setOptions(options);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
			resampler.setInputFormat(data);
			data = Filter.useFilter(data, resampler);
		} catch (Exception e) {
			
		}
		
		// Discretize any continuous variables
		Discretize discretizeFilter = new Discretize();
		try {
			discretizeFilter.setInputFormat(data);
			data = Filter.useFilter(data, discretizeFilter);
		} catch (Exception e) {
			System.out.println("Could not set input format of discretize filter.");
		}
		
		
		

		// Build a naive bayes classifer for the data
		
		NaiveBayesSimple simpleNaiveBayes = new NaiveBayesSimple();
		try {
			simpleNaiveBayes.buildClassifier(data);
		} catch (Exception e) {
			System.out.println("Cannot build classifer");
		}
		
		// Print out the results
		Evaluation evaluation = null;
		try {
			evaluation = new Evaluation(data);
			evaluation.evaluateModel(simpleNaiveBayes, data);
			System.out.println(evaluation.toSummaryString());
			System.out.println(evaluation.toMatrixString());
		} catch (Exception e) {
			System.out.println("Error outputting evaluation");
		}
		

	}
}
