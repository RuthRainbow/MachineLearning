import java.io.File;
import weka.classifiers.meta.MetaCost;
import weka.classifiers.trees.J48;
import weka.core.converters.ArffLoader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;

public class DecisionTreeClassifier {
    public static void main(String[] args) {
        
        // Set the data sources to the user supplied training and test data
        Instances trainingData = null;
        Instances testData = null;
        try {
            ArffLoader trainingLoader = new ArffLoader();
            trainingLoader.setSource(new File(args[0]));
            trainingData = trainingLoader.getDataSet();
            ArffLoader testLoader = new ArffLoader();
            testLoader.setSource(new File(args[1]));
            testData = testLoader.getDataSet();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        trainingData.setClassIndex(0);
        testData.setClassIndex(0);
       
        Discretize discretize = new Discretize(); // new instance of filter
        try {
            discretize.setInputFormat(trainingData); // inform filter about dataset
            // **AFTER** setting options
            trainingData = Filter.useFilter(trainingData, discretize); // apply filter
            testData= Filter.useFilter(testData, discretize); // apply filter
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } // set options
       
        SMOTE smote = new SMOTE(); // new instance of filter
        try {
            smote.setInputFormat(trainingData); // inform filter about dataset
            // **AFTER** setting options
            trainingData = Filter.useFilter(trainingData, smote); // apply filter
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } // set options
       
        String[] options = new String[1];
        options[0] = "-R";
        J48 tree = new J48(); // new instance of tree
        try {
            tree.setOptions(options); // set the options
        }
        catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
       
        // Set up cost matrix for the classifier
        CostMatrix costMatrix = new CostMatrix(2);
        costMatrix.setElement(0, 0, 0);
        costMatrix.setElement(0, 1, 1);
        costMatrix.setElement(1, 0, 10;
        costMatrix.setElement(1, 1, 0);

        MetaCost metaCost = new MetaCost();
        metaCost.setClassifier(tree);
        metaCost.setCostMatrix(costMatrix);
       
        try {
            metaCost.buildClassifier(trainingData);
        } catch (Exception e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace();
        }
       
        // evaluate classifier and print some statistics
        Evaluation eval;
        try {
            eval = new Evaluation(trainingData);
            eval.evaluateModel(metaCost, testData);
            System.out.println(eval.toSummaryString("\nResults\n\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }       
    }   
}
