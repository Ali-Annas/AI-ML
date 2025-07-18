package ie.atu.sw;

import java.awt.FlowLayout;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import javax.swing.Timer;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JFrame;

public class GameWindow implements KeyListener, ActionListener{
	private GameView view;
	private BufferedWriter dataWriter;
	private Timer straightLogger;
	private long lastKeyPressTime = 0;
	
	public GameWindow() throws Exception {
		view = new GameView(false); //Use true to get the plane to fly in autopilot mode...
		init();
		loadSprites();
		// Initialize data writer for training data collection
		try {
			dataWriter = new BufferedWriter(new FileWriter("training_data.csv", true));
			System.out.println("Data logging started. File: training_data.csv");
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// Initialize timer for automatic STRAIGHT action logging (every 2 seconds)
		straightLogger = new Timer(2000, this); // 2000ms = 2 seconds
		straightLogger.start();
	}

	
	/*
	 * Build and display the GUI. 
	 */
	public void init() throws Exception {
	 	var f = new JFrame("ATU - B.Sc. in Software Development");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.addKeyListener(this);
        f.getContentPane().setLayout(new FlowLayout());
        f.add(view);
        f.setSize(1000,1000);
        f.setLocation(100,100);
        f.pack();
        f.setVisible(true);
        f.requestFocus(); // Ensure the window has focus to receive key events
	}
	
	
	/*
	 * Load the sprite graphics from the image directory
	 */
	public void loadSprites() throws Exception {
		var player = new Sprite("Player", 2,  "images/0.png", "images/1.png");
		view.setSprite(player);
		
		var explosion = new Sprite("Explosion", 7,  "images/2.png", 
				"images/3.png", "images/4.png", "images/5.png", 
				"images/6.png", "images/7.png", "images/8.png");
		view.setDyingSprite(explosion);
	}
	
	
	/*
	 * KEYBOARD OPTIONS
	 * ----------------
	 * UP Arrow Key: 	Moves plane up
	 * DOWN Arrow Key: 	Moves plane down
	 * S:				Resets and restarts the game
	 * 
	 * Maybe consider adding options for "start sampling" and "end
	 * sampling"
	 * 
	 */
	public void keyPressed(KeyEvent e) {
		if (e.getKeyCode() == KeyEvent.VK_S) {	//Press "S" to restart
			view.reset(); 						//Reset the view and bail out			
			return;
		}
		
		// Update last key press time
		lastKeyPressTime = System.currentTimeMillis();
		
		int action = 0;
		if (e.getKeyCode() == KeyEvent.VK_UP) {
			action = -1;
		} else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
			action = 1;
		} else {
			// For other keys, log STRAIGHT action
			action = 0;
		}

		view.move(action);						//Move one step

		// --- Data Logging (only during manual mode) ---
		if (!view.isAutoMode()) {
			try {
				double[] features = view.sample();
				String line = Arrays.toString(features).replaceAll("[\\[\\] ]", "") + "," + action;
				dataWriter.write(line);
				dataWriter.newLine();
				dataWriter.flush();
				
				// Show logging message
				String actionName = action == -1 ? "UP" : (action == 1 ? "DOWN" : "STRAIGHT");
				System.out.println("Data logged: action=" + action + " (" + actionName + "), features=" + features.length);
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}

    public void keyReleased(KeyEvent e) {} 		//Ignore
	public void keyTyped(KeyEvent e) {} 		//Ignore

    // Call this method when closing the game to close the writer
    public void closeWriter() {
        try {
            if (dataWriter != null) dataWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    // Timer action for automatic STRAIGHT action logging
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == straightLogger) {
            // Only log during manual mode
            if (!view.isAutoMode()) {
                // Check if no key was pressed recently (within last 1.5 seconds)
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastKeyPressTime > 1500) { // No key pressed for 1.5 seconds
                    // Log STRAIGHT action automatically
                    try {
                        double[] features = view.sample();
                        String line = Arrays.toString(features).replaceAll("[\\[\\] ]", "") + ",0"; // 0 = STRAIGHT
                        dataWriter.write(line);
                        dataWriter.newLine();
                        dataWriter.flush();
                        
                        // Show logging message
                        System.out.println("Data logged: action=0 (STRAIGHT), features=" + features.length + " (auto)");
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }
    }
    

}