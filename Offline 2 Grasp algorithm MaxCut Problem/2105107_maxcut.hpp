#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

class Edge {
public:
    int u, v, weight;
    Edge(int u, int v, int weight) : u(u), v(v), weight(weight) {}
};

class Graph {
public:
    int V;
    vector<Edge> edges;

    Graph(int vertices) : V(vertices) {}

    void addEdge(int u, int v, int weight) {
        edges.emplace_back(u, v, weight);
    }

    Edge getMaxWeightEdge() const {
        return *max_element(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.weight < b.weight;
        });
    }
};

int computeCutWeight(const Graph& g, const unordered_set<int>& X, const unordered_set<int>& Y) {
    int weight = 0;
    for (const Edge& e : g.edges) {
        if ((X.count(e.u) && Y.count(e.v)) || (X.count(e.v) && Y.count(e.u)))
            weight += e.weight;
    }
    return weight;
}

// Randomized Max-Cut
double randomizedMaxCut(const Graph& g, int n) {
    int totalCutWeight = 0;
    for (int i = 0; i < n; i++) {
        vector<int> partition(g.V + 1);
        for (int v = 1; v <= g.V; v++) {
            partition[v] = rand() % 2;
        }
        int cutWeight = 0;
        for (const Edge& e : g.edges) {
            if (partition[e.u] != partition[e.v])
                cutWeight += e.weight;
        }
        totalCutWeight += cutWeight;
    }
    return static_cast<double>(totalCutWeight) / n;
}

// Greedy Max-Cut
pair<unordered_set<int>, unordered_set<int>> greedyMaxCut(const Graph& g) {
    unordered_set<int> X, Y;
    vector<bool> assigned(g.V + 1, false);
    Edge maxEdge = g.getMaxWeightEdge();
    X.insert(maxEdge.u);
    Y.insert(maxEdge.v);
    assigned[maxEdge.u] = true;
    assigned[maxEdge.v] = true;

    for (int z = 1; z <= g.V; z++) {
        if (assigned[z]) continue;
        int wX = 0, wY = 0;
        for (const Edge& e : g.edges) {
            if (e.u == z && Y.count(e.v)) wX += e.weight;
            if (e.v == z && Y.count(e.u)) wX += e.weight;
            if (e.u == z && X.count(e.v)) wY += e.weight;
            if (e.v == z && X.count(e.u)) wY += e.weight;
        }
        if (wX > wY) X.insert(z);
        else Y.insert(z);
        assigned[z] = true;
    }
    return {X, Y};
}

// Semi-Greedy Max-Cut
pair<unordered_set<int>, unordered_set<int>> semiGreedyMaxCut(const Graph& g, double alpha) {
    unordered_set<int> X, Y;
    vector<bool> assigned(g.V + 1, false);

    // Step 1: Start by selecting the edge with the maximum weight
    Edge maxEdge = g.getMaxWeightEdge();
    X.insert(maxEdge.u);
    Y.insert(maxEdge.v);
    assigned[maxEdge.u] = true;
    assigned[maxEdge.v] = true;

    // Step 2: Create an adjacency list representation of the graph
    vector<vector<pair<int, int>>> adj(g.V + 1);
    for (const Edge& e : g.edges) {
        adj[e.u].emplace_back(e.v, e.weight);
        adj[e.v].emplace_back(e.u, e.weight);
    }

    // Step 3: Set up unassigned vertices
    unordered_set<int> unassigned;
    for (int i = 1; i <= g.V; i++) {
        if (!assigned[i]) {
            unassigned.insert(i);
        }
    }

    // Step 4: Continue until all vertices are assigned
    while (!unassigned.empty()) {
        vector<tuple<int, double, double>> candidates;
        double w_min = 1e9;
        double w_max = -1e9;

        // Step 5: Calculate the greedy value for each unassigned vertex
        for (int v : unassigned) {
            double sigmaX = 0, sigmaY = 0;

            for (const auto& [neighbor, weight] : adj[v]) {
                if (X.count(neighbor)) sigmaY += weight;
                else if (Y.count(neighbor)) sigmaX += weight;
            }

            double greedy_val = max(sigmaX, sigmaY);
            w_min = min(w_min, min(sigmaX, sigmaY));
            w_max = max(w_max, greedy_val);

            candidates.push_back({v, sigmaX, sigmaY});
        }

        // Step 6: Calculate mu based on w_min and w_max
        double mu = w_min + alpha * (w_max - w_min);

        // Step 7: Build the Restricted Candidate List (RCL)
        vector<tuple<int, double, double>> RCL;
        for (auto &[v, sigmaX, sigmaY] : candidates) {
            double greedy_val = max(sigmaX, sigmaY);
            if (greedy_val >= mu) {
                RCL.push_back({v, sigmaX, sigmaY});
            }
        }

        // Step 8: If RCL is empty, select the first candidate
        if (RCL.empty()) {
            RCL = candidates;
        }

        // Step 9: Randomly select a candidate from RCL
        int idx = rand() % RCL.size();
        auto [v, sigmaX, sigmaY] = RCL[idx];

        // Step 10: Assign the chosen vertex to the set that maximizes the cut weight
        if (sigmaX > sigmaY) {
            X.insert(v);
        } else {
            Y.insert(v);
        }

        // Step 11: Mark the vertex as assigned
        assigned[v] = true;
        unassigned.erase(v);
    }

    return {X, Y};
}

// Local Search Max-Cut
pair<pair<unordered_set<int>, unordered_set<int>>, int> localSearchMaxCut(const Graph& g, unordered_set<int> X, unordered_set<int> Y) {
    bool improved = true;
    int iterations = 0;

    while (improved) {
        iterations++;
        improved = false;

        // Try moving each vertex to the other unordered_set
        for (int v = 1; v <= g.V; v++) {
            if (X.count(v) == 0 && Y.count(v) == 0) continue;  // Skip if the vertex is not in either unordered_set

            long long sigmaS = 0, sigmaSbar = 0;
            for (const Edge& e : g.edges) {
                if (e.u == v) {
                    if (Y.count(e.v)) sigmaS += e.weight;  // Vertex `v` is connected to `Y`
                    else sigmaSbar += e.weight;             // Vertex `v` is connected to `X`
                }
                else if (e.v == v) {
                    if (Y.count(e.u)) sigmaS += e.weight;  // Vertex `v` is connected to `Y`
                    else sigmaSbar += e.weight;             // Vertex `v` is connected to `X`
                }
            }

            // Move vertex from X to Y if the condition holds
            if (X.count(v) && (sigmaSbar - sigmaS > 0)) {
                // Remove from X and add to Y
                X.erase(v);
                Y.insert(v);
                improved = true;
            }
            // Move vertex from Y to X if the condition holds
            else if (Y.count(v) && (sigmaS - sigmaSbar > 0)) {
                // Remove from Y and add to X
                Y.erase(v);
                X.insert(v);
                improved = true;
            }
        }
    }

    return {{X, Y}, iterations};
}

// GRASP Max-Cut
pair<unordered_set<int>, unordered_set<int>> GRASP(const Graph& g, int maxIterations, double alpha, int earlyStopThreshold = 10) {
    unordered_set<int> bestX, bestY;
    int bestWeight = -1;
    int noImprovementCount = 0;

    for (int i = 0; i < maxIterations; ++i) {
        auto [X, Y] = semiGreedyMaxCut(g, alpha);

        // Unpack the result from localSearchMaxCut
        auto [partition, iter] = localSearchMaxCut(g, X, Y);
        auto [newX, newY] = partition;

        int weight = computeCutWeight(g, newX, newY);
        
        // Check if the solution has improved
        if (weight > bestWeight) {
            bestWeight = weight;
            bestX = newX;
            bestY = newY;
            noImprovementCount = 0;  // Reset if we found a better solution
        } else {
            noImprovementCount++;
        }

        // Early stopping condition: If no improvement over a certain number of iterations, stop
        if (noImprovementCount >= earlyStopThreshold) {
            break;
        }
    }

    return {bestX, bestY};
}
