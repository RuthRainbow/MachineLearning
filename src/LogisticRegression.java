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

		Discretize discreteFilter = new Discretize();
		try {
			discreteFilter.setInputFormat(instances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		int numAttrs = instances.numAttributes();

		for (int i = 1; i < numAttrs; i++) {
			Attribute thisAttr = instances.attribute(i);
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
		
		int count = 0;
		Iterator<Instance> iter = instances.iterator();
		while (iter.hasNext()) {
			try {
				double val = logistic.classifyInstance(iter.next());
				if (val == 1.0) {
					count++;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		System.out.println(count + " out of " + instances.numInstances() + " default");
	}
}
