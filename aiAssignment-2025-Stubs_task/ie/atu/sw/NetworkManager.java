package ie.atu.sw;

import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import java.io.File;

public class NetworkManager {
    
    // Save the trained network to a file
    public static void saveNetwork(BasicNetwork network, String filename) {
        try {
            EncogDirectoryPersistence.saveObject(new File(filename), network);
            System.out.println("Network saved to: " + filename);
        } catch (Exception e) {
            System.out.println("Error saving network: " + e.getMessage());
        }
    }
    
    // Load the trained network from a file
    public static BasicNetwork loadNetwork(String filename) {
        try {
            BasicNetwork network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(filename));
            System.out.println("Network loaded from: " + filename);
            return network;
        } catch (Exception e) {
            System.out.println("Error loading network: " + e.getMessage());
            return null;
        }
    }
} 