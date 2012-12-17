import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;



public class LogisticRegression {
	
	public static void main(String[] args) {
		
		DataSource source = null;
		try {
			source = new DataSource("Data.arff");
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
		
		System.out.println(instances.firstInstance());
	}
}