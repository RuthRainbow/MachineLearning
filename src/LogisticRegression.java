import java.util.Iterator;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.classifiers.functions.SimpleLogistic;

public class LogisticRegression {

	public static void main(String[] args) {

		DataSource source = null;
		try {
			source = new DataSource("credit1.arff");
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
		instances.setClassIndex(instances.numAttributes() - 11);

		System.out.println(instances.firstInstance());

		// Discretise the continuous attributes
		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(instances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		try {
			Filter.useFilter(instances, discreteFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println("after filtering " + instances.firstInstance());
		
		SimpleLogistic logistic = new SimpleLogistic();
		try {
			logistic.buildClassifier(instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("after classifying " + instances.firstInstance());
		
		int truePos = 0;
		int falseNeg = 0;
		int trueNeg = 0;
		int falsePos = 0;
		
		Iterator<Instance> iter = instances.iterator();
		while (iter.hasNext()) {
			try {
				Instance thisInst = iter.next();
				double val = logistic.classifyInstance(thisInst);
				
				Attribute actualAttr = thisInst.attribute(instances.numAttributes() - 11);
				double actualVal = thisInst.value(actualAttr);
				if (val == 1.0 && actualVal == 1.0) {
					truePos++;
				} else if (val == 1.0 && actualVal == 0.0) {
					falseNeg++;
				} else if (val == 0.0 && actualVal == 0.0) {
					trueNeg++;
				} else {
					falsePos++;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		System.out.println("correct ones : " + truePos + " correct zeros : " + trueNeg + 
				" false pos : " + falseNeg + " false neg : " + falsePos);
	}
}
