
#include "2105107_maxcut.hpp"


int main() {
    srand(time(0));
    auto start = chrono::high_resolution_clock::now();

    freopen("in.txt", "r", stdin);
    freopen("out.txt", "w", stdout);
    int V, E;
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    Graph g(V);
    cout << "Enter edges (u v weight):" << endl;
    for (int i = 0; i < E; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        g.addEdge(u, v, w);
    }

    // Randomized
    int trials = 1000;
    double avgRandom = randomizedMaxCut(g, trials);
    cout << "\nRandomized Max-Cut average weight (over " << trials << " trials): " << avgRandom << endl;

    // Greedy
    auto [GX, GY] = greedyMaxCut(g);
    int greedyWeight = computeCutWeight(g, GX, GY);
    cout << "\nGreedy Max-Cut Partition:\nunordered_set X: "; for (int v : GX) cout << v << " "; cout << "\nunordered_set Y: "; for (int v : GY) cout << v << " ";
    cout << "\nGreedy Cut Weight: " << greedyWeight << endl;

    // Semi-Greedy
    double alpha = 0.75;
    cout<<"\n Semi-greedy starts"<<endl;
    auto [SX, SY] = semiGreedyMaxCut(g, alpha);
    int semiGreedyWeight = computeCutWeight(g, SX, SY);
    cout << "\nSemi-Greedy Max-Cut (α = " << alpha << "):\nunordered_set X: "; for (int v : SX) cout << v << " "; cout << "\nunordered_set Y: "; for (int v : SY) cout << v << " ";
    cout << "\nSemi-Greedy Cut Weight: " << semiGreedyWeight << endl;

    // Local Search
    auto [partition, iter] = localSearchMaxCut(g, GX, GY);
    auto [SX_final, SY_final] = partition;
    int localImprovedWeight = computeCutWeight(g, SX_final, SY_final);
    cout << "\nLocal search Max-Cut (α = " << alpha << "):\nunordered_set X: "; for (int v : SX_final) cout << v << " "; cout << "\nunordered_set Y: "; for (int v : SY_final) cout << v << " ";
    cout << "\nAfter Local Search: " << localImprovedWeight << endl;

    // GRASP
    cout << "\nGRASP Max-Cut (α = " << alpha << "):\n";
    int maxIterations = 50;
    auto [GRASP_X, GRASP_Y] = GRASP(g, maxIterations, alpha);
    int graspWeight = computeCutWeight(g, GRASP_X, GRASP_Y);
    cout << "\nGRASP Max-Cut Partition:\nunordered_set X: "; for (int v : GRASP_X) cout << v << " "; cout << "\nunordered_set Y: "; for (int v : GRASP_Y) cout << v << " ";
    cout << "\nGRASP Cut Weight: " << graspWeight << endl;

    // Calculate the total time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double>(end - start);
    cout << "Total Time: " << duration.count() / 60 << " minutes" << endl;

    return 0;
}
