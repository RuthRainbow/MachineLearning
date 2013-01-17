import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MetaCost;

public class KNN
{
	public static void main(String[] args) {
		// Read in the test and training data filepaths from the command line

		int argsLength = args.length;
		if (argsLength < 2) {
			System.out.println("Please specific a training data file and a test data file");
			System.exit(0);
		}

		System.out.println("KNN: Discrete+Smote+metaCost");
		
		String testFilename = args[1];
		String trainingFilename = args[0];
		
		Instances trainingInstances = readInstances(trainingFilename);
		Instances testInstances = readInstances(testFilename);
		
		final int k = 15;
		
		//Set the test data's class index as the "SeriousDlqin2yrs" attribute
		testInstances.setClassIndex(0);
		
		//Prepare to use Discretize and SMOTE
		Discretize discretize = new Discretize();
		SMOTE smote = new SMOTE();
		
		try
		{
			//Apple discretize and then SMOTE to training data
			smote.setInputFormat(trainingInstances);
			discretize.setInputFormat(trainingInstances);
			trainingInstances = Filter.useFilter(trainingInstances, discretize);
			trainingInstances = Filter.useFilter(trainingInstances, smote);
			//Apple discretize to our test data
			discretize.setInputFormat(testInstances);
			testInstances = Filter.useFilter(testInstances, discretize);
		}
		catch (Exception e2)
		{
			System.err.println("Could not randomize data.");
			System.exit(-1);
		}
		
		//Prepare IBk
		IBk ibk = new IBk();
		ibk.setKNN(k);
		ibk.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));

		//Create cost matrix for use with metacost
		CostMatrix costMatrix = new CostMatrix(2);
		costMatrix.setElement(0, 0, 0);
		costMatrix.setElement(0, 1, 1);
		costMatrix.setElement(1, 0, 8);
		costMatrix.setElement(1, 1, 0);
		
		//Prepare metacost, building it with our test data and IBk
		MetaCost metacost = new MetaCost();
		metacost.setCostMatrix(costMatrix);
		metacost.setClassifier(ibk);
		try
		{metacost.buildClassifier(trainingInstances);}
		catch (Exception e)
		{
			System.err.println("Could not build metaCost Classifier.");
			System.exit(-1);
		}
		System.out.println("Built the metacost classifier from ibk.");

		System.out.println("Evaluating "+testInstances.numInstances()+" instances.");
		Evaluation evaluation = null;
		try
		{
			//Evaluate our test data using metacost
			evaluation = new Evaluation(testInstances);
			evaluation.evaluateModel(metacost, testInstances);
			//Print useful information
			System.out.println(evaluation.toSummaryString());
			System.out.println(evaluation.toMatrixString());
			System.out.println(evaluation.toClassDetailsString());
		}
		catch (Exception e)
		{System.err.println("Error outputting evaluation.");}
	}
	
	// Read instances from the given file name and return these
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
