import java.util.Iterator;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemoveFolds;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.RemoveRange;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MetaCost;

//Not yet producing the correct results... I'm looking into it.
public class Knn
{
	public static void main(String[] args) {
		final int k = 15;
		final double samplePercent = 50.0;
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

		Resample resample = new Resample();
		Instances resampledInstances = null;
		try
		{
			resample.setInputFormat(instances);
			resample.setBiasToUniformClass(0.5);
			resample.setSampleSizePercent(samplePercent);
			resampledInstances = Filter.useFilter(instances, resample);
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
			randomize.setInputFormat(resampledInstances);
			randomizedInstances = Filter.useFilter(resampledInstances, randomize);
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
			System.out.println("Built the ibk classifier with "+ibk.getNumTraining()+" training instances.");
			
			CostMatrix costMatrix = new CostMatrix(2);
			costMatrix.setElement(0, 0, 0);
			costMatrix.setElement(0, 1, 1);
			costMatrix.setElement(1, 0, 8);
			costMatrix.setElement(1, 1, 0);
			
			MetaCost metacost = new MetaCost();
			metacost.setCostMatrix(costMatrix);
			metacost.setClassifier(ibk);
			try
			{metacost.buildClassifier(trainingInstances);}
			catch (Exception e1)
			{
				System.err.println("Could not build MetaCost Classifier for fold "+i+".");
				System.exit(-1);
			}
			System.out.println("Built the metacost classifier from ibk.");

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
