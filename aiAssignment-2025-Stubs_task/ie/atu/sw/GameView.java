package ie.atu.sw;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.concurrent.ThreadLocalRandom.current;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Arrays;
import java.util.LinkedList;

import javax.swing.JPanel;
import javax.swing.Timer;

// Add Encog imports for neural network
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;

import ie.atu.sw.NetworkManager;

public class GameView extends JPanel implements ActionListener{
	//Some constants
	private static final long serialVersionUID	= 1L;
	private static final int MODEL_WIDTH 		= 30;
	private static final int MODEL_HEIGHT 		= 20;
	private static final int SCALING_FACTOR 	= 30;
	
	private static final int MIN_TOP 			= 2;
	private static final int MIN_BOTTOM 		= 18;
	private static final int PLAYER_COLUMN 		= 15;
	private static final int TIMER_INTERVAL 	= 100;
	private static final int FAST_INTERVAL 	= 80;  // Faster for easy tunnels
	private static final int SLOW_INTERVAL 	= 150; // Slower for difficult tunnels
	
	private static final byte ONE_SET 			=  1;
	private static final byte ZERO_SET 			=  0;

	/*
	 * The 30x20 game grid is implemented using a linked list of 
	 * 30 elements, where each element contains a byte[] of size 20. 
	 */
	private LinkedList<byte[]> model = new LinkedList<>();

	//These two variables are used by the cavern generator. 
	private int prevTop = MIN_TOP;
	private int prevBot = MIN_BOTTOM;
	
	//Once the timer stops, the game is over
	private Timer timer;
	private long time;
	
	private int playerRow = 11;
	private int index = MODEL_WIDTH - 1; //Start generating at the end
	private Dimension dim;
	
	//Some fonts for the UI display
	private Font font = new Font ("Dialog", Font.BOLD, 50);
	private Font over = new Font ("Dialog", Font.BOLD, 100);

	//The player and a sprite for an exploding plane
	private Sprite sprite;
	private Sprite dyingSprite;
	
	private boolean auto;
	
	// Neural network for autopilot (currently disabled for rule-based approach)
	// private BasicNetwork neuralNetwork;
	
	// Neural network for autopilot (disabled for now)
	// private BasicNetwork neuralNetwork;
	
	// State machine for autopilot to prevent loops
	private int lastMove = 0; // -1=UP, 0=STRAIGHT, 1=DOWN
	private int consecutiveMoves = 0; // Count consecutive same moves
	private int lastPlayerRow = 11; // Track last position

	public GameView(boolean auto) throws Exception{
		this.auto = auto; //Use the autopilot
		setBackground(Color.LIGHT_GRAY);
		setDoubleBuffered(true);
		
		//Creates a viewing area of 900 x 600 pixels
		dim = new Dimension(MODEL_WIDTH * SCALING_FACTOR, MODEL_HEIGHT * SCALING_FACTOR);
    	super.setPreferredSize(dim);
    	super.setMinimumSize(dim);
    	super.setMaximumSize(dim);
		
    	initModel();
    	
		timer = new Timer(TIMER_INTERVAL, this); //Timer calls actionPerformed() every second
		timer.start();
		
		// Load neural network if in autopilot mode (disabled for now)
		// if (auto) {
		// 	loadNeuralNetwork();
		// }
	}
	
	// Load the trained neural network (disabled for now)
	/*
	private void loadNeuralNetwork() {
		try {
			// For now, we'll create a simple network here
			// In a real implementation, you'd load a saved network from file
			neuralNetwork = createSimpleNetwork();
			System.out.println("Neural network loaded for autopilot mode");
		} catch (Exception e) {
			System.out.println("Error loading neural network: " + e.getMessage());
			neuralNetwork = null;
		}
	}
	
	// Load the trained network from file
	private BasicNetwork createSimpleNetwork() {
		try {
			// Load the trained network from the saved file
			return NetworkManager.loadNetwork("trained_network.eg");
		} catch (Exception e) {
			System.out.println("Could not load trained network: " + e.getMessage());
			return null;
		}
	}
	*/
	
	//Build our game grid
	private void initModel() {
		for (int i = 0; i < MODEL_WIDTH; i++) {
			model.add(new byte[MODEL_HEIGHT]);
		}
	}
	
	public void setSprite(Sprite s) {
		this.sprite = s;
	}
	
	public void setDyingSprite(Sprite s) {
		this.dyingSprite = s;
	}
	
	//Called every second by actionPerformed(). Paint methods are usually ugly.
	public void paintComponent(Graphics g) {
        super.paintComponent(g);
        var g2 = (Graphics2D)g;
        
        g2.setColor(Color.WHITE);
        g2.fillRect(0, 0, dim.width, dim.height);
        
        int x1 = 0, y1 = 0;
        for (int x = 0; x < MODEL_WIDTH; x++) {
        	for (int y = 0; y < MODEL_HEIGHT; y++){  
    			x1 = x * SCALING_FACTOR;
        		y1 = y * SCALING_FACTOR;

        		if (model.get(x)[y] != 0) {
            		if (y == playerRow && x == PLAYER_COLUMN) {
            			timer.stop(); //Crash...
            		}
            		g2.setColor(Color.BLACK);
            		g2.fillRect(x1, y1, SCALING_FACTOR, SCALING_FACTOR);
        		}
        		
        		if (x == PLAYER_COLUMN && y == playerRow) {
        			if (timer.isRunning()) {
            			g2.drawImage(sprite.getNext(), x1, y1, null);
        			}else {
            			g2.drawImage(dyingSprite.getNext(), x1, y1, null);
        			}
        			
        		}
        	}
        }
        
        /*
         * Not pretty, but good enough for this project... The compiler will
         * tidy up and optimise all of the arithmetics with constants below.
         */
        g2.setFont(font);
        g2.setColor(Color.RED);
        g2.fillRect(1 * SCALING_FACTOR, 15 * SCALING_FACTOR, 400, 3 * SCALING_FACTOR);
        g2.setColor(Color.WHITE);
        g2.drawString("Time: " + (int)(time * (TIMER_INTERVAL/1000.0d)) + "s", 1 * SCALING_FACTOR + 10, (15 * SCALING_FACTOR) + (2 * SCALING_FACTOR));
        
        if (!timer.isRunning()) {
			g2.setFont(over);
			g2.setColor(Color.RED);
			g2.drawString("Game Over!", MODEL_WIDTH / 5 * SCALING_FACTOR, MODEL_HEIGHT / 2* SCALING_FACTOR);
			System.out.println("[CRASH] Game Over! Plane crashed at row=" + playerRow + ", col=" + PLAYER_COLUMN);
        }
	}

	//Move the plane up or down
	public void move(int step) {
		playerRow += step;
		// Keep player within bounds
		if (playerRow < 0) playerRow = 0;
		if (playerRow >= MODEL_HEIGHT) playerRow = MODEL_HEIGHT - 1;
	}
	
	
	/*
	 * ----------
	 * AUTOPILOT!
	 * ----------
	 * The following implementation uses neural network predictions instead of random movements.
	 * If neural network is not available, it falls back to random movement.
	 *  
	 */
	private void autoMove() {
		int playerCol = PLAYER_COLUMN;
		int playerRow = this.playerRow;
		int lookAhead = 20; // Increased lookahead for better planning
		int bestMove = 0; // 0=STRAIGHT, -1=UP, 1=DOWN
		int moveSteps = 1; // Default move one step

		// Calculate current tunnel bounds
		int topWall = -1, bottomWall = MODEL_HEIGHT;
		for (int row = 0; row < MODEL_HEIGHT; row++) {
			if (model.get(playerCol)[row] == 1) {
				if (topWall == -1) topWall = row;
				bottomWall = row;
			}
		}
		
		// Calculate tunnel width and center
		int tunnelWidth = (bottomWall - topWall);
		int tunnelCenter = (topWall + bottomWall) / 2;
		
		System.out.println("[DEBUG] Player at col=" + playerCol + ", row=" + playerRow + 
						 ", topWall=" + topWall + ", bottomWall=" + bottomWall + 
						 ", tunnelWidth=" + tunnelWidth + ", center=" + tunnelCenter);

		// Edge detection: If at top or bottom, always move away from edge
		if (playerRow == 0) {
			System.out.println("[EDGE] At top edge! Forcing DOWN.");
			if (model.get(playerCol)[playerRow + 1] == 0) move(1);
			else System.out.println("[BLOCKED] Can't move DOWN, wall below!");
			return;
		}
		if (playerRow == MODEL_HEIGHT - 1) {
			System.out.println("[EDGE] At bottom edge! Forcing UP.");
			if (model.get(playerCol)[playerRow - 1] == 0) move(-1);
			else System.out.println("[BLOCKED] Can't move UP, wall above!");
			return;
		}

		// Analyze tunnel ahead for better path planning
		int[] tunnelWidths = new int[lookAhead];
		int[] tunnelCenters = new int[lookAhead];
		boolean[] hasObstacle = new boolean[lookAhead];
		
		for (int i = 0; i < lookAhead && (playerCol + i + 1) < MODEL_WIDTH; i++) {
			int col = playerCol + i + 1;
			int top = -1, bottom = MODEL_HEIGHT;
			
			for (int row = 0; row < MODEL_HEIGHT; row++) {
				if (model.get(col)[row] == 1) {
					if (top == -1) top = row;
					bottom = row;
				}
			}
			
			if (top != -1) {
				tunnelWidths[i] = bottom - top;
				tunnelCenters[i] = (top + bottom) / 2;
				hasObstacle[i] = (model.get(col)[playerRow] == 1);
			} else {
				tunnelWidths[i] = MODEL_HEIGHT;
				tunnelCenters[i] = MODEL_HEIGHT / 2;
				hasObstacle[i] = false;
			}
		}
		
		// Check for immediate collision
		boolean willHitWall = hasObstacle[0];
		
		// Enhanced collision detection: Look ahead for potential collisions
		boolean hasEscapeRoute = checkEscapeRoutes(playerCol, playerRow, lookAhead);
		
		if (willHitWall || !hasEscapeRoute) {
			// PANIC MODE: Wall ahead or no escape route!
			System.out.println("[PANIC] " + (willHitWall ? "Wall ahead detected!" : "No escape route ahead!"));
			
			// Find best escape route
			int upSpace = 0, downSpace = 0;
			for (int row = playerRow - 1; row >= 0; row--) {
				if (model.get(playerCol)[row] == 0) {
					upSpace++;
				} else {
					break;
				}
			}
			for (int row = playerRow + 1; row < MODEL_HEIGHT; row++) {
				if (model.get(playerCol)[row] == 0) {
					downSpace++;
				} else {
					break;
				}
			}
			
			System.out.println("[DEBUG] Escape routes - upSpace=" + upSpace + ", downSpace=" + downSpace);
			
			// Try double moves if possible
			if (upSpace >= 2 && model.get(playerCol)[playerRow - 2] == 0) {
				move(-2);
				System.out.println("[PANIC] Double UP move!");
				return;
			} else if (downSpace >= 2 && model.get(playerCol)[playerRow + 2] == 0) {
				move(2);
				System.out.println("[PANIC] Double DOWN move!");
				return;
			}
			// Choose best escape direction
			if (upSpace == 0 && downSpace > 0) {
				if (model.get(playerCol)[playerRow + 1] == 0) {
					bestMove = 1;
					System.out.println("[PANIC] Only DOWN available!");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move DOWN! Staying STRAIGHT");
				}
			} else if (downSpace == 0 && upSpace > 0) {
				if (model.get(playerCol)[playerRow - 1] == 0) {
					bestMove = -1;
					System.out.println("[PANIC] Only UP available!");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move UP! Staying STRAIGHT");
				}
			} else if (upSpace > downSpace) {
				if (model.get(playerCol)[playerRow - 1] == 0) {
					bestMove = -1;
					System.out.println("[PANIC] Moving UP (more space)");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move UP! Staying STRAIGHT");
				}
			} else if (downSpace > 0) {
				if (model.get(playerCol)[playerRow + 1] == 0) {
					bestMove = 1;
					System.out.println("[PANIC] Moving DOWN (more space)");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move DOWN! Staying STRAIGHT");
				}
			} else {
				// Both up and down are blocked, slide along the wall or search for nearest open cell
				System.out.println("[PANIC] Both up and down blocked! Searching for nearest open cell...");
				int nearest = -1, minDist = MODEL_HEIGHT;
				for (int r = 0; r < MODEL_HEIGHT; r++) {
					if (model.get(playerCol)[r] == 0 && Math.abs(r - playerRow) < minDist) {
						nearest = r;
						minDist = Math.abs(r - playerRow);
					}
				}
				if (nearest != -1) {
					if (nearest < playerRow) bestMove = -1;
					else if (nearest > playerRow) bestMove = 1;
					else bestMove = 0;
					System.out.println("[PANIC] Sliding toward open cell at row=" + nearest);
				} else {
					bestMove = 0;
					System.out.println("[PANIC] No open cells! Staying STRAIGHT");
				}
			}
		} else {
			// NORMAL MODE: No immediate collision
			
			// Find optimal path by looking ahead
			int targetRow = playerRow;
			double bestScore = Double.MAX_VALUE;
			
			for (int i = 0; i < lookAhead && (playerCol + i + 1) < MODEL_WIDTH; i++) {
				if (tunnelWidths[i] > 0) {
					// Calculate score based on distance to center and tunnel width
					int distanceToCenter = Math.abs(playerRow - tunnelCenters[i]);
					double widthFactor = 1.0 / Math.max(1, tunnelWidths[i]); // Prefer wider tunnels
					double score = distanceToCenter + (widthFactor * 10); // Weight width factor
					
					if (score < bestScore) {
						bestScore = score;
						targetRow = tunnelCenters[i];
					}
				}
			}
			
			// Enhanced path planning: Find the best route through the tunnel
			targetRow = findOptimalPath(playerCol, playerRow, lookAhead);
			
			// Check if we're too close to walls
			int distanceToTop = playerRow - topWall;
			int distanceToBottom = bottomWall - playerRow;
			int minSafeDistance = 2;
			
			if (distanceToTop <= minSafeDistance) {
				if (model.get(playerCol)[playerRow + 1] == 0) {
					bestMove = 1;
					System.out.println("[SAFETY] Too close to top wall, moving DOWN");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move DOWN! Staying STRAIGHT");
				}
			} else if (distanceToBottom <= minSafeDistance) {
				if (model.get(playerCol)[playerRow - 1] == 0) {
					bestMove = -1;
					System.out.println("[SAFETY] Too close to bottom wall, moving UP");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move UP! Staying STRAIGHT");
				}
			} else if (playerRow < targetRow - 1) {
				if (model.get(playerCol)[playerRow + 1] == 0) {
					bestMove = 1;
					System.out.println("Autopilot: DOWN (moving to center) [targetRow=" + targetRow + "]");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move DOWN! Staying STRAIGHT");
				}
			} else if (playerRow > targetRow + 1) {
				if (model.get(playerCol)[playerRow - 1] == 0) {
					bestMove = -1;
					System.out.println("Autopilot: UP (moving to center) [targetRow=" + targetRow + "]");
				} else {
					bestMove = 0;
					System.out.println("[BLOCKED] Can't move UP! Staying STRAIGHT");
				}
			} else {
				bestMove = 0;
				System.out.println("Autopilot: STRAIGHT (in good position) [targetRow=" + targetRow + "]");
			}
		}
		
		// Execute the move
		for (int i = 0; i < moveSteps; i++) move(bestMove);
		
		// Update state machine
		updateAutopilotState(bestMove);
		
		// Adjust game speed based on tunnel complexity
		adjustGameSpeed();
	}
	
	// Update autopilot state to detect and prevent loops
	private void updateAutopilotState(int move) {
		if (move == lastMove) {
			consecutiveMoves++;
		} else {
			consecutiveMoves = 1;
			lastMove = move;
		}
		
		// If we're stuck in a loop (same move 5+ times or same position 10+ times)
		if (consecutiveMoves > 5 || (playerRow == lastPlayerRow && consecutiveMoves > 3)) {
			System.out.println("[LOOP-DETECTED] Breaking out of repetitive pattern!");
			// Force a different move
			if (move == 0) {
				// If we were going straight, try moving up or down
				if (model.get(PLAYER_COLUMN)[playerRow - 1] == 0) {
					move(-1);
					System.out.println("[LOOP-BREAK] Forcing UP");
				} else if (model.get(PLAYER_COLUMN)[playerRow + 1] == 0) {
					move(1);
					System.out.println("[LOOP-BREAK] Forcing DOWN");
				}
			}
			consecutiveMoves = 0;
		}
		
		lastPlayerRow = playerRow;
	}

	// Find optimal path through the tunnel using depth-limited lookahead
	private int findOptimalPath(int playerCol, int playerRow, int lookAhead) {
		int bestTargetRow = playerRow;
		double bestScore = Double.MAX_VALUE;
		int searchDepth = Math.min(5, lookAhead); // Simulate up to 5 steps ahead
		
		for (int target = Math.max(0, playerRow - 3); target <= Math.min(MODEL_HEIGHT - 1, playerRow + 3); target++) {
			double score = simulatePath(playerCol, playerRow, target, searchDepth);
			if (score < bestScore) {
				bestScore = score;
				bestTargetRow = target;
			}
		}
		return bestTargetRow;
	}

	// Simulate a path for a given target row and return a score (lower is better)
	private double simulatePath(int col, int row, int targetRow, int depth) {
		if (depth == 0 || col + 1 >= MODEL_WIDTH) return 0;
		int nextCol = col + 1;
		int bestNextRow = row;
		double bestScore = Double.MAX_VALUE;
		for (int move = -1; move <= 1; move++) {
			int newRow = row + move;
			if (newRow < 0 || newRow >= MODEL_HEIGHT) continue;
			if (model.get(nextCol)[newRow] == 1) continue; // Wall
			// Score: distance to target + penalty for being near wall
			int top = -1, bottom = MODEL_HEIGHT;
			for (int r = 0; r < MODEL_HEIGHT; r++) {
				if (model.get(nextCol)[r] == 1) {
					if (top == -1) top = r;
					bottom = r;
				}
			}
			int tunnelWidth = (top != -1) ? (bottom - top) : MODEL_HEIGHT;
			int center = (top != -1) ? (top + bottom) / 2 : MODEL_HEIGHT / 2;
			int distToCenter = Math.abs(newRow - center);
			int distToTarget = Math.abs(newRow - targetRow);
			double widthPenalty = (tunnelWidth < 8) ? (8 - tunnelWidth) * 2 : 0;
			double score = distToTarget + distToCenter + widthPenalty;
			score += simulatePath(nextCol, newRow, targetRow, depth - 1);
			if (score < bestScore) {
				bestScore = score;
				bestNextRow = newRow;
			}
		}
		return bestScore;
	}
	
	// Check if there are viable escape routes ahead
	private boolean checkEscapeRoutes(int playerCol, int playerRow, int lookAhead) {
		// Look ahead to see if there are any viable paths
		for (int i = 1; i <= lookAhead && (playerCol + i) < MODEL_WIDTH; i++) {
			int col = playerCol + i;
			int top = -1, bottom = MODEL_HEIGHT;
			
			// Find tunnel bounds at this column
			for (int row = 0; row < MODEL_HEIGHT; row++) {
				if (model.get(col)[row] == 1) {
					if (top == -1) top = row;
					bottom = row;
				}
			}
			
			if (top != -1) {
				int tunnelWidth = bottom - top;
				// If we find a wide enough tunnel ahead, we have an escape route
				if (tunnelWidth >= 4) {
					return true;
				}
			}
		}
		
		// No viable escape routes found
		return false;
	}

	// Adjust game speed based on tunnel complexity
	private void adjustGameSpeed() {
		if (!auto) return; // Only adjust in autopilot mode
		
		// Calculate current tunnel complexity
		int complexity = calculateTunnelComplexity();
		
		int newInterval;
		if (complexity > 8) {
			newInterval = SLOW_INTERVAL; // Slow down for complex tunnels
		} else if (complexity < 3) {
			newInterval = FAST_INTERVAL; // Speed up for simple tunnels
		} else {
			newInterval = TIMER_INTERVAL; // Normal speed
		}
		
		// Only change if significantly different
		if (Math.abs(timer.getDelay() - newInterval) > 10) {
			timer.setDelay(newInterval);
			System.out.println("[SPEED] Adjusted to " + newInterval + "ms (complexity: " + complexity + ")");
		}
	}

	// Calculate tunnel complexity based on width variations ahead
	private int calculateTunnelComplexity() {
		int complexity = 0;
		int playerCol = PLAYER_COLUMN;
		
		// Look ahead 10 columns and count width changes
		for (int i = 1; i <= 10 && (playerCol + i) < MODEL_WIDTH; i++) {
			int col = playerCol + i;
			int top = -1, bottom = MODEL_HEIGHT;
			
			for (int row = 0; row < MODEL_HEIGHT; row++) {
				if (model.get(col)[row] == 1) {
					if (top == -1) top = row;
					bottom = row;
				}
			}
			
			if (top != -1) {
				int width = bottom - top;
				if (width < 8) complexity++; // Narrow tunnels are complex
				if (width < 6) complexity++; // Very narrow tunnels are very complex
			}
		}
		
		return complexity;
	}
	
	
	//Called every second by the timer 
	public void actionPerformed(ActionEvent e) {
		time++; //Update our timer
		this.repaint(); //Repaint the cavern
		
		//Update the next index to generate
		index++;
		index = (index == MODEL_WIDTH) ? 0 : index;
		
		generateNext(); //Generate the next part of the cave
		if (auto) {
			autoMove();
		}
		
		/*
		 * STRAIGHT action logging is now handled by GameWindow timer
		 * This ensures proper CSV file writing
		 */
	}
	
	
	/*
	 * Generate the next layer of the cavern. Use the linked list to
	 * move the current head element to the tail and then randomly
	 * decide whether to increase or decrease the cavern. 
	 */
	private void generateNext() {
		var next = model.pollFirst(); 
		model.addLast(next); //Move the head to the tail
		Arrays.fill(next, ONE_SET); //Fill everything in
		
		// Fairer tunnel generation
		var minspace = 8; // Increased minimum tunnel width for fairness
		int maxStep = 1; // Limit tunnel movement per step
		
		// Limit how much the tunnel can move up or down per step
		int topStep = current().nextBoolean() ? 1 : -1;
		int botStep = current().nextBoolean() ? 1 : -1;
		prevTop += Math.max(-maxStep, Math.min(maxStep, topStep));
		prevBot += Math.max(-maxStep, Math.min(maxStep, botStep));
		
		// Prevent tunnel from going out of bounds
		prevTop = max(MIN_TOP, min(prevTop, prevBot - minspace));
		prevBot = min(MIN_BOTTOM, max(prevBot, prevTop + minspace));
		
		//Fill in the array with the carved area
		Arrays.fill(next, prevTop, prevBot, ZERO_SET);
	}
	
	
	/*
	 * Use this method to get a snapshot of the 30x20 matrix of values
	 * that make up the game grid. The grid is flatmapped into a single
	 * dimension double array... (somewhat) ready to be used by a neural 
	 * net. You can experiment around with how much of this you actually
	 * will need. The plane is always somehere in column PLAYER_COLUMN
	 * and you probably do not need any of the columns behind this. You
	 * can consider all of the columns ahead of PLAYER_COLUMN as your
	 * horizon and this value can be reduced to save space and time if
	 * needed, e.g. just look 1, 2 or 3 columns ahead. 
	 * 
	 * You may also want to track the last player movement, i.e.
	 * up, down or no change. Depending on how you design your neural
	 * network, you may also want to label the data as either okay or 
	 * dead. Alternatively, the label might be the movement (up, down
	 * or straight). 
	 *  
	 */
	public double[] sample() {
		var vector = new double[MODEL_WIDTH * MODEL_HEIGHT];
		var index = 0;
		
		for (byte[] bm : model) {
			for (byte b : bm) {
				vector[index] = b;
				index++;
			}
		}
		return vector;
	}
	
	public void reset() {
		time = 0;
		playerRow = 11;
		index = MODEL_WIDTH - 1;
		prevTop = MIN_TOP;
		prevBot = MIN_BOTTOM;
		
		// Clear the model
		model.clear();
		initModel();
		
		// Restart the timer
		if (timer != null) {
			timer.start();
		}
	}
	
	// Method to check if autopilot mode is enabled
	public boolean isAutoMode() {
		return auto;
	}
	

}