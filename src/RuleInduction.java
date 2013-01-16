import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.classifiers.rules.Prism;
import weka.filters.supervised.instance.Resample;

public class RuleInduction {

  public static void main(String[] args) {
		
		//String test = args[0];
		String test = "testData.arff";
		
		DataSource testData = null;
		DataSource source = null;
		try {
			source = new DataSource("credit1.arff");
			testData = new DataSource(test);
		} catch (Exception e) {
			System.out.println("could not find file");
			e.printStackTrace();
		}

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

		System.out.println(testInstances.instance(2));
		
		// Replace the missing values
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
		
		Resample resample = new Resample();
		try {
			resample.setBiasToUniformClass(0.5);
			resample.setInputFormat(instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		try {
			instances = Filter.useFilter(instances, resample);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Discretise the continuous attributes
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
		
		System.out.println("after discretize: " + testInstances.instance(0));
		System.out.println(instances.instance(0));

		Prism prism = new Prism();
		try {
			prism.buildClassifier(instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		int truePos = 0;
		int falseNeg = 0;
		int trueNeg = 0;
		int falsePos = 0;

		int numInstances = testInstances.numInstances();
		for (int i = 0; i < numInstances; i++) {
			try {
				Instance thisInst = testInstances.instance(i);
				double val = prism.classifyInstance(thisInst);

				Attribute actualAttr = thisInst.attribute(0);
				double actualVal = thisInst.value(actualAttr);
				if (val == 1.0 && actualVal == 1.0) {
					trueNeg++;
				} else if (val == 1.0 && actualVal == 0.0) {
					falseNeg++;
				} else if (val == 0.0 && actualVal == 0.0) {
					truePos++;
				} else {
					falsePos++;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		System.out.println("correct ones : " + trueNeg + " correct zeros : " + truePos +
				" false pos : " + falsePos + " false neg : " + falseNeg);
	}
}
