
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.instance.Randomize;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;

//Not yet producing the correct results... I'm looking into it.
public class Knn
{
	public static void main(String[] args) {
		final int k = 15;
		final double samplePercent = 1.0;
		final int folds = 10;
		final int seed = 827634;

		DataSource source = null;
		
		try
		{source = new DataSource("./testdata.arff");}
		catch (Exception e)
		{
			System.err.println("Could not find file.");
			System.exit(-1);
		}

		Instances instances = null;
		try
		{instances = source.getDataSet();}
		catch (Exception e)
		{
			System.err.println("Could not get dataset from file.");
			System.exit(-1);
		}

		// Set the class index as the "SeriousDlqin2yrs" attribute
		instances.setClassIndex(0);
		System.out.println("Smoting small...");
		SMOTE sample = new SMOTE();
		Instances sampledInstances = null;
		try
		{
			sample.setInputFormat(instances);
			//sample.setClassValue("0");
			sample.setPercentage(samplePercent);
			sample.setRandomSeed(seed);
			sampledInstances = Filter.useFilter(instances, sample);
		}
		catch (Exception e1)
		{
			System.err.println("Could not resample instances for training.");
			System.exit(-1);
		}		
		
		
		Randomize randomize = new Randomize();
		Instances randomizedInstances = null;
		
		randomize.setRandomSeed(seed);
		
		try
		{
			randomize.setInputFormat(sampledInstances);
			randomizedInstances = Filter.useFilter(sampledInstances, randomize);
		}
		catch (Exception e2)
		{
			System.err.println("Could not randomize data.");
			System.exit(-1);
		}
		

		Instances trainingInstances = null;
		Instances testInstances = null;
		for(int i = 1; i < folds; i++)
		{

			trainingInstances = randomizedInstances.trainCV(folds, i-1);
			testInstances = randomizedInstances.testCV(folds, i-1);
			
			System.out.println("Fold "+i+" of "+folds+": ");
			
			IBk ibk = new IBk();
			ibk.setKNN(k);
			ibk.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
			try
			{ibk.buildClassifier(trainingInstances);}
			catch (Exception e)
			{
				System.err.println("Could not build IBk Classifier for fold "+i+".");
				System.exit(-1);
			}
			System.out.println("Built the classifier with "+ibk.getNumTraining()+" training instances.");

			System.out.println("Evaluating "+testInstances.numInstances()+" instances.");
			Evaluation evaluation = null;
			try
			{
				evaluation = new Evaluation(testInstances);
				evaluation.evaluateModel(ibk, testInstances);
				System.out.println(evaluation.toSummaryString());
				System.out.println(evaluation.toMatrixString());
				System.out.println(evaluation.toClassDetailsString());
			}
			catch (Exception e)
			{System.err.println("Error outputting evaluation for fold "+i+".");}
		}
	}
}
