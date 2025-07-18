import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.*;
import java.util.*;

public class TrainEncogNN {
    public static void main(String[] args) throws Exception {
        System.out.println("=== NEURAL NETWORK TRAINING STARTED ===");
        
        // 1. Read CSV data
        System.out.println("\n1. Loading training data...");
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("training_data.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        }
        
        System.out.println("Total data samples: " + lines.size());
        
        int featureSize = lines.get(0).split(",").length - 1;
        System.out.println("Feature size: " + featureSize);
        
        double[][] X = new double[lines.size()][featureSize];
        double[][] y = new double[lines.size()][3]; // 3 classes: up, straight, down
        
        // Count actions for analysis
        int upCount = 0, straightCount = 0, downCount = 0;
        
        for (int i = 0; i < lines.size(); i++) {
            String[] parts = lines.get(i).split(",");
            for (int j = 0; j < featureSize; j++) {
                X[i][j] = Double.parseDouble(parts[j]);
            }
            int label = Integer.parseInt(parts[featureSize]) + 1; // -1,0,1 -> 0,1,2
            y[i][label] = 1.0;
            
            // Count actions
            if (label == 0) upCount++;
            else if (label == 1) straightCount++;
            else if (label == 2) downCount++;
        }
        
        System.out.println("\n=== DATA DISTRIBUTION ===");
        System.out.println("UP actions (-1): " + upCount + " (" + String.format("%.1f", (upCount * 100.0 / lines.size())) + "%)");
        System.out.println("STRAIGHT actions (0): " + straightCount + " (" + String.format("%.1f", (straightCount * 100.0 / lines.size())) + "%)");
        System.out.println("DOWN actions (1): " + downCount + " (" + String.format("%.1f", (downCount * 100.0 / lines.size())) + "%)");
        
        if (straightCount < lines.size() * 0.1) {
            System.out.println("⚠️  WARNING: Very few STRAIGHT actions! This may cause poor autopilot performance.");
        }

        // 2. Create dataset
        System.out.println("\n2. Creating training dataset...");
        MLDataSet dataset = new BasicMLDataSet(X, y);

        // 3. Build neural network
        System.out.println("\n3. Building neural network architecture...");
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, featureSize));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 32));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 16));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 3));
        network.getStructure().finalizeStructure();
        network.reset();
        
        System.out.println("Network architecture: " + featureSize + " -> 32 -> 16 -> 3");
        System.out.println("Total weights: " + network.getStructure().getFlat().getWeights().length);

        // 4. Train the network
        System.out.println("\n4. Training neural network...");
        System.out.println("Training parameters:");
        System.out.println("- Learning rate: Adaptive (Resilient Propagation)");
        System.out.println("- Max epochs: 100");
        System.out.println("- Target error: 0.01");
        
        ResilientPropagation train = new ResilientPropagation(network, dataset);
        int epoch = 0;
        double bestError = Double.MAX_VALUE;
        
        System.out.println("\n=== TRAINING PROGRESS ===");
        System.out.println("Epoch\tError\t\tImprovement");
        System.out.println("-----\t-----\t\t-----------");
        
        do {
            double previousError = train.getError();
            train.iteration();
            epoch++;
            double currentError = train.getError();
            double improvement = previousError - currentError;
            
            if (epoch % 5 == 0) {
                System.out.printf("%d\t%.6f\t%.6f\n", epoch, currentError, improvement);
            }
            
            if (currentError < bestError) {
                bestError = currentError;
            }
            
        } while (train.getError() > 0.01 && epoch < 100);
        train.finishTraining();
        
        System.out.println("\n=== TRAINING COMPLETE ===");
        System.out.println("Final error: " + train.getError());
        System.out.println("Best error: " + bestError);
        System.out.println("Total epochs: " + epoch);

        // 5. Test predictions on training data
        System.out.println("\n5. Testing predictions on training data...");
        int correctPredictions = 0;
        int totalPredictions = 0;
        int[] classPredictions = new int[3]; // Count predictions for each class
        
        for (int i = 0; i < Math.min(100, X.length); i++) { // Test first 100 samples
            MLData input = new BasicMLData(X[i]);
            MLData output = network.compute(input);
            
            int pred = 0;
            double max = output.getData(0);
            for (int j = 1; j < 3; j++) {
                if (output.getData(j) > max) {
                    max = output.getData(j);
                    pred = j;
                }
            }
            
            // Find actual label
            int actual = 0;
            for (int j = 0; j < 3; j++) {
                if (y[i][j] == 1.0) {
                    actual = j;
                    break;
                }
            }
            
            if (pred == actual) {
                correctPredictions++;
            }
            classPredictions[pred]++;
            totalPredictions++;
        }
        
        double accuracy = (correctPredictions * 100.0) / totalPredictions;
        System.out.println("\n=== TRAINING ACCURACY ===");
        System.out.println("Accuracy: " + String.format("%.2f", accuracy) + "% (" + correctPredictions + "/" + totalPredictions + ")");
        System.out.println("Prediction distribution:");
        System.out.println("- UP (-1): " + classPredictions[0] + " predictions");
        System.out.println("- STRAIGHT (0): " + classPredictions[1] + " predictions");
        System.out.println("- DOWN (1): " + classPredictions[2] + " predictions");

        // 6. Save the trained network
        System.out.println("\n6. Saving trained network...");
        NetworkManager.saveNetwork(network, "trained_network.eg");

        // 7. Test prediction on first sample
        System.out.println("\n7. Final test prediction:");
        MLData input = new BasicMLData(X[0]);
        MLData output = network.compute(input);
        int pred = 0;
        double max = output.getData(0);
        for (int i = 1; i < 3; i++) {
            if (output.getData(i) > max) {
                max = output.getData(i);
                pred = i;
            }
        }
        
        String[] actions = {"UP", "STRAIGHT", "DOWN"};
        System.out.println("Sample prediction: " + actions[pred] + " (value: " + (pred - 1) + ")");
        System.out.println("Output values: [" + String.format("%.3f", output.getData(0)) + 
                         ", " + String.format("%.3f", output.getData(1)) + 
                         ", " + String.format("%.3f", output.getData(2)) + "]");
        
        System.out.println("\n=== TRAINING COMPLETE ===");
        System.out.println("Network saved! Ready to use in game.");
        System.out.println("Recommendation: Test autopilot and collect more data if needed.");
    }
} 