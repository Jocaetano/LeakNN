package leaknn;

import java.io.File;
import java.util.Arrays;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class LeakNN {

    public void run(String[] args) {
        try {

            File irisFile = new File("DataSet_learn.csv");

            // Define the format of the data file.
            // This area will change, depending on the columns and 
            // format of the file that you are trying to model.
            VersatileDataSource source = new CSVDataSource(irisFile, false, CSVFormat.DECIMAL_POINT);
            VersatileMLDataSet data = new VersatileMLDataSet(source);
            data.defineInput(data.defineSourceColumn("m", 0, ColumnType.continuous));
            data.defineInput(data.defineSourceColumn("p", 1, ColumnType.continuous));
            data.defineInput(data.defineSourceColumn("s", 2, ColumnType.continuous));
            data.defineInput(data.defineSourceColumn("v1", 3, ColumnType.continuous));
            data.defineInput(data.defineSourceColumn("v2", 4, ColumnType.continuous));
            data.defineInput(data.defineSourceColumn("v3", 5, ColumnType.continuous));

            // Define the column that we are trying to predict.
            data.defineOutput(data.defineSourceColumn("vazamento", 6, ColumnType.continuous));
            data.defineOutput(data.defineSourceColumn("nvazamento", 7, ColumnType.continuous));
           // ColumnDefinition outputColumn = data.defineSourceColumn("nvazamento", 7, ColumnType.continuous);

            // Analyze the data, determine the min/max/mean/sd of every column.
            data.analyze();

            // Map the prediction column to the output of the model, and all
            // other columns to the input.
//            data.defineSingleOutputOthersInput(outputColumn);

            // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
            // You could also other model types, such as:
            // MLMethodFactory.SVM:  Support Vector Machine (SVM)
            // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
            // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
            // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
            EncogModel model = new EncogModel(data);
            model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);

            // Send any output to the console.
            model.setReport(new ConsoleStatusReportable());

            // Now normalize the data.  Encog will automatically determine the correct normalization
            // type based on the model you chose in the last step.
            data.normalize();

            // Hold back some data for a final validation.
            // Shuffle the data into a random ordering.
            // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
            model.holdBackValidation(0.3, true, 1001);

            // Choose whatever is the default training type for this model.
            model.selectTrainingType(data);

            // Use a 5-fold cross-validated train.  Return the best method found.
            MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);

            // Display the training and validation errors.
            System.out.println("Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
            System.out.println("Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

            // Display our normalization parameters.
            NormalizationHelper helper = data.getNormHelper();
            System.out.println(helper.toString());

            // Display the final model.
            System.out.println("Final model: " + bestMethod);

            // Loop over the entire, original, dataset and feed it through the model.
            // This also shows how you would process new data, that was not part of your
            // training set.  You do not need to retrain, simply use the NormalizationHelper
            // class.  After you train, you can save the NormalizationHelper to later
            // normalize and denormalize your data.
            irisFile = new File("DataSet_test.csv");
            ReadCSV csv = new ReadCSV(irisFile, false, CSVFormat.DECIMAL_POINT);
            String[] line = new String[6];
            MLData input = helper.allocateInputVector();

            while (csv.next()) {
                StringBuilder result = new StringBuilder();
                line[0] = csv.get(0);
                line[1] = csv.get(1);
                line[2] = csv.get(2);
                line[3] = csv.get(3);
                line[4] = csv.get(4);
                line[5] = csv.get(5);
                String correct = csv.get(6);
                helper.normalizeInputVector(line, input.getData(), false);
                MLData output = bestMethod.compute(input);
                String irisChosen = helper.denormalizeOutputVectorToString(output)[0];
                String irisChosen2 = helper.denormalizeOutputVectorToString(output)[1];

                result.append(Arrays.toString(line));
                result.append(" -> predicted: ");
                result.append(irisChosen);
                result.append(", ");
                result.append(irisChosen2);
                result.append("(correct: ");
                result.append(correct);
                result.append(")");

                System.out.println(result.toString());
            }

            Encog.getInstance().shutdown();

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        LeakNN prg = new LeakNN();
        prg.run(args);
    }
}
