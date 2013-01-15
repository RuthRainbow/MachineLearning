import weka.classifiers.CostMatrix;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.meta.MetaCost;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;

public class LogisticRegression {

	public static void main(String[] args) {
		
		//String filename = args[0];

		Instances trainingInstances = readInstances("trainingdata");
		
		Resample resample = new Resample();
		try {
			resample.setInputFormat(trainingInstances);
			resample.setBiasToUniformClass(0.5);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		try {
			trainingInstances = Filter.useFilter(trainingInstances, resample);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Discretise the continuous attributes
		Discretize discreteFilter = new Discretize();
		SMOTE smote = new SMOTE();
		try {
			discreteFilter.setInputFormat(trainingInstances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		try {
			trainingInstances = Filter.useFilter(trainingInstances, discreteFilter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/*try {
			smote.setInputFormat(instances);
			smote.setRandomSeed(10);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		try {
			instances = Filter.useFilter(instances, smote);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		*/


		System.out.println("finished filtering");

		SimpleLogistic logistic = new SimpleLogistic();
		try {
			logistic.buildClassifier(trainingInstances);
			//costSense.buildClassifier(instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		CostMatrix costMatrix = new CostMatrix(2);
		costMatrix.setElement(0, 0, 1);
		costMatrix.setElement(0, 1, -0.1);
		costMatrix.setElement(1, 0, -0.2);
		costMatrix.setElement(1, 1, 1);
		
		MetaCost metaCost = new MetaCost();
		metaCost.setClassifier(logistic);
		metaCost.setCostMatrix(costMatrix);
		try {
			metaCost.buildClassifier(trainingInstances);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		System.out.println("built the classifier");
		
		//Instances testInstances = readInstances("testdata");

		int truePos = 0;
		int falsePos = 0;
		int trueNeg = 0;
		int falseNeg = 0;

		int numInstances = trainingInstances.numInstances();
		for (int i = 0; i < numInstances; i++) {
			try {
				Instance thisInst = trainingInstances.instance(i);
				double val = logistic.classifyInstance(thisInst);

				Attribute actualAttr = thisInst.attribute(0);
				double actualVal = thisInst.value(actualAttr);
				if (val == 1.0 && actualVal == 1.0) {
					truePos++;
				} else if (val == 1.0 && actualVal == 0.0) {
					falsePos++;
				} else if (val == 0.0 && actualVal == 0.0) {
					trueNeg++;
				} else {
					falseNeg++;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		System.out.println("correct ones : " + truePos + " correct zeros : " + trueNeg + 
				" false neg : " + falsePos + " false pos : " + falseNeg);
		
		System.out.println(truePos + "   " + falseNeg);
		System.out.println(falsePos + "    " + trueNeg);
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
