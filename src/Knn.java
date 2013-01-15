import java.util.Iterator;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.lazy.IBk;

//Not yet producing the correct results... I'm looking into it.
public class Knn
{
	public static void main(String[] args) {
		final int k = 1;
		final double samplePercent = 100.0;

		DataSource source = null;
		
		try
		{source = new DataSource("./credit1.arff");}
		catch (Exception e)
		{
			System.out.println("Could not find file.");
			System.exit(-1);
		}

		Instances instances = null;
		try
		{instances = source.getDataSet();}
		catch (Exception e)
		{
			System.out.println("Could not get dataset from file.");
			System.exit(-1);
		}

		// Set the class index as the "SeriousDlqin2yrs" attribute
		instances.setClassIndex(0);

		Discretize discretizeFilter = new Discretize();
		Instances filteredInstances = null;
		
		try {
			discretizeFilter.setInputFormat(instances);
			filteredInstances = Filter.useFilter(instances, discretizeFilter);
		} catch (Exception e) {
			System.out.println("Could not set input format of discretize filter.");
		}
		//filteredInstances = instances;

		Resample resample = new Resample();
		Instances trainingInstances = null;
		try
		{
			resample.setInputFormat(instances);
			resample.setBiasToUniformClass(0.5);
			resample.setSampleSizePercent(samplePercent);
			trainingInstances = Filter.useFilter(instances, resample);
		}
		catch (Exception e1)
		{
			System.out.println("Could not resample instances for training.");
			System.exit(-1);
		}

		IBk ibk = new IBk();
		ibk.setKNN(k);
		try
		{ibk.buildClassifier(trainingInstances);}
		catch (Exception e)
		{
			System.out.println("Could not build IBk Classifier.");
			System.exit(-1);
		}

		System.out.println("Built the classifier with "+(ibk.getNumTraining())+" training instances.");

		int truePos = 0;
		int falseNeg = 0;
		int trueNeg = 0;
		int falsePos = 0;

		System.out.print("Progress: 0%");
		Iterator<Instance> iter = filteredInstances.iterator();
		int total = filteredInstances.size();
		int count = 0;
		int last = 0;
		while (iter.hasNext()) {
			try {
				Instance thisInst = iter.next();
				double val = ibk.classifyInstance(thisInst);

				Attribute actualAttr = thisInst.attribute(0);
				double actualVal = thisInst.value(actualAttr);
				if(val == actualVal)
				{
					if(actualVal == 1.0)
					{truePos++;}
					else
					{trueNeg++;}
				}
				else
				{
					if(actualVal == 1.0)
					{falsePos++;}
					else
					{falseNeg++;}
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			count++;
			if(Math.floor((100*count)/total) != last)
			{
				last = (int)Math.floor((100*count)/total);
				System.out.print("\rProgress: "+last+"%");
			}
		}
		System.out.println();

		System.out.println("True Positive: " + truePos + " (" + Math.round((100.0*truePos)/(truePos+falseNeg)) +"%)");
		System.out.println("True Negative: " + trueNeg + " (" + Math.round((100.0*trueNeg)/(trueNeg+falsePos)) +"%)");
		System.out.println("False Positive: " + falseNeg + " (" + Math.round((100.0*falseNeg)/(truePos+falseNeg)) +"%)");
		System.out.println("False Negative: " + falsePos + " (" + Math.round((100.0*falsePos)/(trueNeg+falsePos)) +"%)");
		System.out.println("Total Classifications: "+ count);
	}
}
