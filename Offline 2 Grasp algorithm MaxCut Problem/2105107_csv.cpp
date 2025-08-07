#include "2105107_maxcut.hpp"  // Include the header for graph and algorithm definitions
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>  // For setw to format the output


using namespace std;

// Function to read the graph from a .rud file
Graph readGraphFromFile(const string& filename) {
    ifstream file(filename);
    int V, E;
    file >> V >> E;

    Graph g(V);
    for (int i = 0; i < E; i++) {
        int u, v, weight;
        file >> u >> v >> weight;
        g.addEdge(u, v, weight);
    }
    return g;
}



//hardcode known best
int giveKnownbest(int graphNum) {
    switch (graphNum) {
    //     unordered_map<string, int> knownBest = {
    // {"G1", 12078}, {"G2", 12084}, {"G3", 12077}, {"G11", 627}, {"G12", 621}, {"G13", 645}, {"G14", 3187}, {"G15", 3169}, {"G16", 3172}, {"G22", 14123}, {"G23", 14129}, {"G24", 14131}, {"G32", 1560}, {"G33", 1537}, {"G34", 1541}, {"G35", 8000}, {"G36", 7996}, {"G37", 8009}, {"G43", 7027}, {"G44", 7022}, {"G45", 7020}, {"G48", 6000}, {"G49", 6000}, {"G50", 5988}};
        case 1: return 12078;
        case 2: return 12084;
        case 3: return 12077;
        case 11: return 627;
        //complete all
        case 12: return 621;
        case 13: return 645;
        case 14: return 3187;
        case 15: return 3169;
        case 16: return 3172;
        case 22: return 14123;
        case 23: return 14129;
        case 24: return 14131;
        case 32: return 1560;
        case 33: return 1537;
        case 34: return 1541;
        case 35: return 8000;
        case 36: return 7996;
        case 37: return 8009;
        case 43: return 7027;
        case 44: return 7022;
        case 45: return 7020;
        case 48: return 6000;
        case 49: return 6000;
        case 50: return 5988;
        default: return -1;  // Unknown graph number

    }
}

// Function to run all algorithms for a given graph
void runAlgorithmsAndStoreResults(Graph& g, int graphNum, double alpha, ofstream& csvFile) {
    // Run Randomized Max-Cut
    double randomizedResult = randomizedMaxCut(g, 1000);  // Number of trials: 1000

    // Run Greedy Max-Cut
    auto [GX, GY] = greedyMaxCut(g);
    int greedyResult = computeCutWeight(g, GX, GY);

    // Run Semi-Greedy Max-Cut
    auto [SX, SY] = semiGreedyMaxCut(g, alpha);
    int semiGreedyResult = computeCutWeight(g, SX, SY);

    // Run Local Search Max-Cut
    auto [partition, iterations] = localSearchMaxCut(g, GX, GY);
    auto [SX_final, SY_final] = partition;
    int localSearchResult = computeCutWeight(g, SX_final, SY_final);

    // Run GRASP Max-Cut
    int max_iterations = 50;  // Number of iterations for GRASP
    auto [GRASP_X, GRASP_Y] = GRASP(g, max_iterations, alpha);  // 50 iterations for GRASP
    int graspResult = computeCutWeight(g, GRASP_X, GRASP_Y);

    // Prepare the result row and write to CSV
    csvFile << "G" << graphNum << "  ,";
    csvFile << g.V << "  ,";  // Number of vertices
    csvFile << g.edges.size() << "  ,";  // Number of edges
    csvFile << randomizedResult << "  ,";  // Simple Randomized or Randomized-1
    csvFile << greedyResult << "  ,";  // Simple Greedy or Greedy-1
    csvFile << semiGreedyResult << "  ,";  // Semi Greedy - 1
    csvFile << iterations << "  ,";  // Simple local or local-1 No. of iterations
    csvFile << localSearchResult << "  ,";  // Average Value for Simple local
    csvFile << max_iterations << "  ,";  // GRASP No. of iterations
    csvFile << graspResult << "\n";  // GRASP-1 Best value
    csvFile << giveKnownbest(graphNum) << "\n";  // Known best value 


    // Print the results to the console in a grid-like format (aligned)
    cout << "Results for Graph G" << graphNum << ":\n";
    cout << setw(25) << left << "Algorithm"
         << setw(25) << "Value" << endl;
    // cout << "----------------------------------------" << endl;

    cout << setw(25) << left << "Randomized Max-Cut"
         << setw(25) << randomizedResult << endl;
    cout << setw(25) << left << "Greedy Max-Cut"
         << setw(25) << greedyResult << endl;
    cout << setw(25) << left << "Semi-Greedy Max-Cut"
         << setw(25) << semiGreedyResult << endl;
    cout << setw(25) << left << "Local Search Max-Cut - Iterations"
         << setw(25) << iterations << endl;
    cout << setw(25) << left << "Local Search Max-Cut - Average Value"
         << setw(25) << localSearchResult << endl;
    cout << setw(25) << left << "GRASP Max-Cut(iter,res)"
         << setw(25) <<max_iterations<<" ,"<< graspResult << endl;
    
}

int main()
{
    srand(time(0));

    // Open CSV file to write the results
    ofstream csvFile("2105107.csv");
    freopen("output.txt", "w", stdout); // Redirect stdout to a file (optional, for debugging purposes)

    cout << "Generating CSV file for Max-Cut results..." << endl;
    // Writing CSV header
    csvFile << "Problem,|V| or n,|E| or m,Simple Randomized or Randomized-1,Simple Greedy or Greedy-1,Semi Greedy - 1,Simple local or local-1 No. of iterations,Average Value, Grasp No. of iterations,Best Value, known best\n";
    auto program_start = chrono::high_resolution_clock::now();
    // Process graph files from g1.rud to g54.rud
    for (int i = 1; i <= 54; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        // Construct the filename for each graph file
        stringstream ss;
        ss << "graph_GRASP/set1/g" << i << ".rud";
        string filename = ss.str();
        cout << "Processing file: " << filename << endl;

        // Read the graph from file
        Graph g = readGraphFromFile(filename);

        // Run algorithms and store results in the CSV file
        runAlgorithmsAndStoreResults(g, i, 0.75, csvFile); // alpha = 0.75 for Semi-Greedy
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration<double>(end - start);

        cout << "Time taken to process file: " << (duration).count() / 60 << " minutes" << endl;
        cout << "----------------------------------------" << endl;
    }

    // Close the CSV file after writing results
    csvFile.close();
    auto program_end = chrono::high_resolution_clock::now();
    auto program_duration = chrono::duration<double>(program_end - program_start);

    cout << "CSV file generated successfully! in " << program_duration.count() / 60 << "minutes" << endl;

    return 0;
}
