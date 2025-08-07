import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

class Node {
    int[][] state;
    int g, h, f;
    Node parent;

    public Node(int[][] state, int g, int h, Node parent) {
        this.state = state;
        this.g = g;
        this.h = h;
        this.f = g + h;
        this.parent = parent;
    }

    void displayNode() {
        System.out.println("State:");
        for (int i = 0; i < state.length; i++) {
            for (int j = 0; j < state[i].length; j++) {
                System.out.print(state[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("g: " + g + "  h: " + h + "  f: " + f);
        System.out.println();
    }
}

interface heuristicFunctions {
    int calculate(Node node, int n);
}

class HammingDistance implements heuristicFunctions {
    public int calculate(Node node, int n) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (node.state[i][j] != 0 && node.state[i][j] != i * n + j + 1) {
                    count++;
                }
            }
        }
        return count;
    }
}

class ManhattanDistance implements heuristicFunctions {
    public int calculate(Node node, int n) {
        int distance = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (node.state[i][j] != 0) {
                    int goalX = (node.state[i][j] - 1) / n;
                    int goalY = (node.state[i][j] - 1) % n;
                    distance += Math.abs(i - goalX) + Math.abs(j - goalY);
                }
            }
        }
        return distance;
    }
}

class EuclideanDistance implements heuristicFunctions {
    public int calculate(Node node, int n) {
        double dist = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (node.state[i][j] != 0) {
                    int goalX = (node.state[i][j] - 1) / n;
                    int goalY = (node.state[i][j] - 1) % n;
                    dist += Math.sqrt(Math.pow(i - goalX, 2) + Math.pow(j - goalY, 2));
                }
            }
        }
        return (int) dist;
    }
}

class LinearConflictHeuristic implements heuristicFunctions {
    ManhattanDistance manhattan = new ManhattanDistance();

    public int calculate(Node node, int n) {
        int man = manhattan.calculate(node, n);
        int conflicts = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (node.state[i][j] != 0 && node.state[i][j] != i * n + j + 1) {
                    int goalX = (node.state[i][j] - 1) / n;
                    int goalY = (node.state[i][j] - 1) % n;
                    if (goalX == i && goalY != j) {
                        conflicts++;
                    }
                }
            }
        }
        return man + 2 * conflicts;
    }
}

class Detecting_Solvable {
    Node initialNode;
    int k;
    int[] states;

    Detecting_Solvable(Node initialNode, int grid_k) {
        this.initialNode = initialNode;
        this.k = grid_k;
        this.states = new int[k * k];
    }

    void make1Darray() {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                states[i * k + j] = initialNode.state[i][j];
            }
        }
    }

    int inversions() {
        make1Darray();
        int count = 0;
        for (int i = 0; i < k * k - 1; i++) {
            for (int j = i + 1; j < k * k; j++) {
                if (states[i] > states[j] && states[i] != 0 && states[j] != 0)
                    count++;
            }
        }
        return count;
    }

    boolean isSolvable() {
        int inv = inversions();
        System.out.println("Number of inversions: " + inv + " for k = " + k + " and initial state: ");
        if (k % 2 != 0) {
            return inv % 2 == 0;
        } else {
            int blankRow = 0;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    if (initialNode.state[i][j] == 0)
                        blankRow = i + 1;
                }
            }
            blankRow++;
            return (blankRow % 2 == 0 && inv % 2 != 0) || (blankRow % 2 != 0 && inv % 2 == 0);
        }
    }
}



class Astar {
    int k;
    heuristicFunctions heuristic;
    int expandedNodesCount = 0;
    int generatedNodesCount = 0; //explored nodes
    PriorityQueue<Node> openList;
    Set<String> closedSet;

    public Astar(int k, Node initial, heuristicFunctions heuristic) {
        this.k = k;
        this.heuristic = heuristic;
        // openList = new PriorityQueue<>(Comparator.comparingInt(n -> n.f));
        openList = new PriorityQueue<>((n1, n2) -> {
            if (n1.f != n2.f) return n1.f - n2.f;
            if (n1.h != n2.h) return n1.h - n2.h;
            return n1.g - n2.g;
        });
        
        closedSet = new HashSet<>();
        initial.h = heuristic.calculate(initial, k);
        initial.f = initial.g + initial.h;
        openList.add(initial);
        generatedNodesCount++;  //explored lists
    }

    public Node solve() {
        while (!openList.isEmpty()) {
            Node current = openList.poll();
            expandedNodesCount++;
            if (isGoal(current))
                return current;
            closedSet.add(stateToString(current.state));

            for (Node neighbor : getNeighbors(current)) {
                String key = stateToString(neighbor.state);
                if (closedSet.contains(key))
                    continue;

                Optional<Node> existing = openList.stream()
                        .filter(n -> Arrays.deepEquals(n.state, neighbor.state))
                        .findFirst();
                if (existing.isPresent()) {
                    if (neighbor.f < existing.get().f) {
                        openList.remove(existing.get());
                        openList.add(neighbor);
                    }
                } else {
                    openList.add(neighbor);
                    generatedNodesCount++;
                }
            }
        }
        return null;
    }

    private List<Node> getNeighbors(Node node) {
        List<Node> neighbors = new ArrayList<>();
        int[][] s = node.state;
        int zx = 0, zy = 0;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                if (s[i][j] == 0) {
                    zx = i;
                    zy = j;
                }

        int[][] dirs = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        for (int[] d : dirs) {
            int nx = zx + d[0], ny = zy + d[1];
            if (nx >= 0 && nx < k && ny >= 0 && ny < k) {
                int[][] newState = new int[k][k];
                for (int i = 0; i < k; i++)
                    newState[i] = s[i].clone();
                newState[zx][zy] = newState[nx][ny];
                newState[nx][ny] = 0;
                Node child = new Node(newState, node.g + 1, 0, node);
                child.h = heuristic.calculate(child, k);
                child.f = child.g + child.h;
                neighbors.add(child);
            }
        }
        return neighbors;
    }

    private boolean isGoal(Node node) {
        int target = 1;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (i == k - 1 && j == k - 1) {
                    if (node.state[i][j] != 0)
                        return false;
                } else if (node.state[i][j] != target++)
                    return false;
            }
        }
        return true;
    }

    private String stateToString(int[][] s) {
        StringBuilder sb = new StringBuilder();
        for (int[] row : s)
            for (int v : row)
                sb.append(v).append(',');
        return sb.toString();
    }

    public List<Node> getPath(Node goal) {
        LinkedList<Node> path = new LinkedList<>();
        for (Node n = goal; n != null; n = n.parent) {
            path.addFirst(n);
        }
        return path;
    }
}



public class Npuzzle_2105107_Offline_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int k = sc.nextInt();
        int[][] initialState = new int[k][k];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                initialState[i][j] = sc.nextInt();

        Node initialNode = new Node(initialState, 0, 0, null);
        Detecting_Solvable solver = new Detecting_Solvable(initialNode, k);
        boolean solvable = solver.isSolvable();
        // System.out.println("Is the puzzle solvable? " + solvable);

        sc.nextLine(); // consume newline

        if (solvable) {
            System.out.print("Choose heuristic (h=Hamming, m=Manhattan, e=Euclidean, l=Linear): ");
            String choice = sc.nextLine().trim().toLowerCase();
            heuristicFunctions heuristic;
            switch (choice) {
                case "h":
                    heuristic = new HammingDistance();
                    break;
                case "m":
                    heuristic = new ManhattanDistance();
                    break;
                case "e":
                    heuristic = new EuclideanDistance();
                    break;
                case "l":
                    heuristic = new LinearConflictHeuristic();
                    break;
                default:
                    System.out.println("Invalid choice");
                    return;
            }

            Astar astar = new Astar(k, initialNode, heuristic);
            Node goal = astar.solve();


            if (goal != null) {
                List<Node> path = astar.getPath(goal);
                System.out.println("Solution found in " + goal.g + " moves.");
                System.out.println("Expanded nodes: " + astar.expandedNodesCount);
                System.out.println("Explored nodes: " + astar.generatedNodesCount);

                try (BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt"))) {
                    writer.write("Minimum number of moves " + goal.g + " moves\n");
                    writer.write("Expanded: " + astar.expandedNodesCount + ", Explored: " + astar.generatedNodesCount
                            + "\n");
                    writer.write("Path states:\n");
                    for (Node n : path) {
                        for (int i = 0; i < k; i++) {
                            for (int j = 0; j < k; j++)
                                writer.write(n.state[i][j] + " ");
                            writer.write("\n");
                        }
                        writer.write("---\n");
                    }
                } catch (IOException e) {
                    System.err.println("Error writing output file: " + e.getMessage());
                }
            } else {
                System.out.println("No solution found.");
            }
        } else {
            System.out.println("The puzzle is not solvable.");
        }
    }
}
