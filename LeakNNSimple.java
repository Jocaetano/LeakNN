package leaknnsimple;

import org.encog.Encog;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.simple.TrainingSetUtil;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class LeakNNSimple {

    public void run(String[] args) {

        final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, "DataSet.csv", false, 6, 2);
        final BasicNetwork network = EncogUtility.simpleFeedForward(6, 2, 0, 2, true);

        final Backpropagation train = new Backpropagation(network, trainingSet, 0.07, 0.02);
        train.setBatchSize(1);

        System.out.println();
        System.out.println("Training Network");
        EncogUtility.trainToError(train, 0.04);

        System.out.println();
        System.out.println("Evaluating Network");
        EncogUtility.evaluate(network, trainingSet);

        double[] input = new double[6];
        input[0] = 3;
        input[1] = 4;
        input[2] = 3.3;
        input[3] = 1.1;
        input[4] = 0.8;
        input[5] = 1.5;
        MLData inputData = new BasicMLData(input);
        double[] output = new double[2];
        output[0] = 1;
        output[1] = 0;
        MLData outputData = new BasicMLData(output);
        // double[] output = new double[2];
        MLDataSet inputRow = new BasicMLDataSet();
        inputRow.add(inputData, outputData);
        System.out.println("");
        EncogUtility.evaluate(network, inputRow);
        System.out.println(network.compute(inputData));
        network.compute(input, output);
        System.out.println(output[0] + " " + output[1]);
        // inputRow.add();

        Encog.getInstance().shutdown();
    }

    public static void main(String[] args) {
        LeakNNSimple prg = new LeakNNSimple();
        prg.run(args);
    }
}
