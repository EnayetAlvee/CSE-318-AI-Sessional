#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>     // For log2
#include <algorithm> // For sort
#include <set>       // For unique values
#include <fstream>   // For file I/O
#include <sstream>   // For stringstream

int criterion; // Global variable to hold the criterion type
// Possible values: 0 for IG, 1 for IGR ,2 for NWIG

// --- 1. Data Structures ---
using namespace std;
// Represents a single row of data
struct DataRow {
    double sepalLength;
    double sepalWidth;
    double petalLength;
    double petalWidth;
    string species; // Target variable
};

// Represents a node in the Decision Tree
enum class FeatureType {
    SepalLength,
    SepalWidth,
    PetalLength,
    PetalWidth,
    Leaf // Indicates a leaf node
};

// Helper function to convert string feature name to enum
FeatureType stringToFeatureType(const string& featureName) {
    if (featureName == "SepalLengthCm") return FeatureType::SepalLength;
    if (featureName == "SepalWidthCm") return FeatureType::SepalWidth;
    if (featureName == "PetalLengthCm") return FeatureType::PetalLength;
    if (featureName == "PetalWidthCm") return FeatureType::PetalWidth;
    return FeatureType::Leaf; // Should not happen for feature names
}

// Helper to get feature value from DataRow based on FeatureType
double getFeatureValue(const DataRow& row, FeatureType feature) {
    switch (feature) {
        case FeatureType::SepalLength: return row.sepalLength;
        case FeatureType::SepalWidth:  return row.sepalWidth;
        case FeatureType::PetalLength: return row.petalLength;
        case FeatureType::PetalWidth:  return row.petalWidth;
        default: return 0.0; // Should not happen for valid features
    }
}


struct Node {
    FeatureType feature;      // Feature used for splitting at this node
    double threshold;         // Split value for continuous features
    map<string, int> class_counts; // Class counts if it's a leaf node
    string majority_class; // Most frequent class in this node/leaf

    Node* left_child;         // Pointer to the left child (value <= threshold)
    Node* right_child;        // Pointer to the right child (value > threshold)
    bool is_leaf;             // True if this is a leaf node

    // Constructor for internal node
    Node(FeatureType f, double t) :
        feature(f), threshold(t), left_child(nullptr), right_child(nullptr), is_leaf(false) {}

    // Constructor for leaf node
    Node(const vector<DataRow>& data) :
        feature(FeatureType::Leaf), threshold(0.0), left_child(nullptr), right_child(nullptr), is_leaf(true) {
        
        // Calculate class counts for the leaf
        for (const auto& row : data) {
            class_counts[row.species]++;
        }

        // Determine majority class for the leaf
        int max_count = -1;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                majority_class = pair.first;
            }
        }
    }

    // Destructor to free memory
    ~Node() {
        delete left_child;
        delete right_child;
    }
};

// --- 2. Core Calculation Functions ---

// Function to calculate Entropy
double calculateEntropy(const vector<DataRow>& data) {
    if (data.empty()) return 0.0;

    map<string, int> class_counts;
    for (const auto& row : data) {
        class_counts[row.species]++;
    }

    double entropy = 0.0;
    int total_count = data.size();

    for (const auto& pair : class_counts) {
        double probability = static_cast<double>(pair.second) / total_count;
        if (probability > 0) { // Avoid log(0)
            entropy -= probability * log2(probability);
        }
    }
    return entropy;
}

// Function to calculate Information Gain for a continuous attribute with a specific split
double calculateInformationGain(const vector<DataRow>& data, FeatureType feature, double split_value) {
    double total_entropy = calculateEntropy(data);

    vector<DataRow> subset_le, subset_gt; // <= split_value and > split_value

    for (const auto& row : data) {
        if (getFeatureValue(row, feature) <= split_value) {
            subset_le.push_back(row);
        } else {
            subset_gt.push_back(row);
        }
    }

    if (subset_le.empty() || subset_gt.empty()) {
        return -1.0; // Invalid split, can't make a division
    }

    double entropy_le = calculateEntropy(subset_le);
    double entropy_gt = calculateEntropy(subset_gt);

    double weighted_entropy = (static_cast<double>(subset_le.size()) / data.size()) * entropy_le +
                              (static_cast<double>(subset_gt.size()) / data.size()) * entropy_gt;

    return total_entropy - weighted_entropy;
}
double calculateInformationGainRatio(const vector<DataRow>& data, FeatureType feature, double split_value) {
    double gain = calculateInformationGain(data, feature, split_value);
    if (gain <= 0.0) return 0.0; // No gain or invalid split

    // Calculate intrinsic value
    double total_count = data.size();
    double subset_le_count = 0.0;
    double subset_gt_count = 0.0;

    for (const auto& row : data) {
        if (getFeatureValue(row, feature) <= split_value) {
            subset_le_count++;
        } else {
            subset_gt_count++;
        }
    }

    double intrinsic_value = 0.0;
    if (subset_le_count > 0) {
        double prob_le = subset_le_count / total_count;
        intrinsic_value -= prob_le * log2(prob_le);
    }
    if (subset_gt_count > 0) {
        double prob_gt = subset_gt_count / total_count;
        intrinsic_value -= prob_gt * log2(prob_gt);
    }

    return intrinsic_value > 0 ? gain / intrinsic_value : 0.0; // Avoid division by zero
}
// Calculate NWIG (Normalized Weighted Information Gain)
// Formula: NWIG(S,A) = (IG(S,A) / log2(k+1)) * (1 - (k-1)/|S|)
double calculateNWIG(const vector<DataRow>& data, FeatureType feature, double split_value) {
    double gain = calculateInformationGain(data, feature, split_value);
    if (gain <= 0.0) return 0.0; // No gain or invalid split

    // Count distinct values for the feature (k)
    set<double> unique_values_set;
    for (const auto& row : data) {
        unique_values_set.insert(getFeatureValue(row, feature));
    }
    double k = static_cast<double>(unique_values_set.size());
    
    // Dataset size |S|
    double dataset_size = static_cast<double>(data.size());
    
    // Avoid division by zero and invalid cases
    if (k <= 1 || dataset_size <= 1) return 0.0;
    
    // Calculate NWIG using the formula:
    // NWIG(S,A) = (IG(S,A) / log2(k+1)) * (1 - (k-1)/|S|)
    double normalization_factor = gain / log2(k + 1.0);
    double penalty_factor = 1.0 - (k - 1.0) / dataset_size;
    
    // Ensure penalty factor is not negative
    if (penalty_factor < 0.0) penalty_factor = 0.0;
    
    return normalization_factor * penalty_factor;
}
// Function to find the best split point for a continuous attribute
pair<double, double> findBestContinuousSplit(const vector<DataRow>& data, FeatureType feature) {
    set<double> unique_values_set;
    for (const auto& row : data) {
        unique_values_set.insert(getFeatureValue(row, feature));
    }

    vector<double> unique_values(unique_values_set.begin(), unique_values_set.end());
    
    if (unique_values.size() < 2) {
        return {0.0, -1.0}; // Cannot split if less than 2 unique values
    }

    double best_gain = -1.0;
    double best_split_value = 0.0;

    // Iterate through midpoints as candidate split points
    for (size_t i = 0; i < unique_values.size() - 1; ++i) {
        double split_value = (unique_values[i] + unique_values[i+1]) / 2.0;
        double current_gain ;
        if(criterion==0) {
            current_gain = calculateInformationGain(data, feature, split_value);
            cout<<"Using IG criterion: " << current_gain << endl;
        } else if(criterion==2) {
            current_gain = calculateNWIG(data, feature, split_value);
            cout<<"Using NWIG criterion: " << current_gain << endl;
        } else if(criterion==1) {
            current_gain = calculateInformationGainRatio(data, feature, split_value);
            cout<<"Using IGR criterion: " << current_gain << endl;
        } else {
            cerr << "Unknown criterion: " << criterion << endl;
            return {0.0, -1.0}; // Invalid criterion
        }
        
        if (current_gain > best_gain) {
            best_gain = current_gain;
            best_split_value = split_value;
        }
    }
    return {best_split_value, best_gain};
}


//

// --- 3. Decision Tree Building ---

Node* buildDecisionTree(vector<DataRow> data, const vector<FeatureType>& features_available, int max_depth, int min_samples_leaf, int current_depth = 0) {
    // Base Case 1: Node is pure
    if (calculateEntropy(data) == 0.0) {
        return new Node(data); // Create a leaf node
    }

    // Base Case 2: Max depth reached
    if (max_depth != -1 && current_depth >= max_depth) {
        return new Node(data); // Create a leaf node
    }

    // Base Case 3: Min samples per leaf not met
    if (data.size() < min_samples_leaf) {
        return new Node(data); // Create a leaf node
    }

    double best_gain = -1.0;
    FeatureType best_feature = FeatureType::Leaf; // Initialize to something indicating no feature
    double best_threshold = 0.0;

    // Find the best split among available features
    for (FeatureType feature : features_available) {
        pair<double, double> split_info = findBestContinuousSplit(data, feature);
        double current_threshold = split_info.first;
        double current_gain = split_info.second;

        if (current_gain > best_gain) {
            best_gain = current_gain;
            best_feature = feature;
            best_threshold = current_threshold;
        }
    }

    // Base Case 4: No significant gain from any split
    if (best_gain <= 0.0) { // Using 0.0 as threshold, can be a small epsilon
        return new Node(data); // Create a leaf node
    }

    // Create a new internal node
    Node* node = new Node(best_feature, best_threshold);

    // Split data for children
    vector<DataRow> left_data, right_data;
    for (const auto& row : data) {
        if (getFeatureValue(row, best_feature) <= best_threshold) {
            left_data.push_back(row);
        } else {
            right_data.push_back(row);
        }
    }

    // Handle empty splits (should be rare if findBestContinuousSplit is robust)
    if (left_data.empty() || right_data.empty()) {
        return new Node(data); // Revert to leaf if split is degenerate
    }

    // Recursively build child trees
    node->left_child = buildDecisionTree(left_data, features_available, max_depth, min_samples_leaf, current_depth + 1);
    node->right_child = buildDecisionTree(right_data, features_available, max_depth, min_samples_leaf, current_depth + 1);

    return node;
}

// --- 4. Prediction Function ---

string predict(Node* node, const DataRow& sample) {
    if (node->is_leaf) {
        return node->majority_class;
    }

    double feature_value = getFeatureValue(sample, node->feature);

    if (feature_value <= node->threshold) {
        return predict(node->left_child, sample);
    } else {
        return predict(node->right_child, sample);
    }
}
int calculateNumberOfNodes(Node* node) {
    if (node == nullptr) return 0;
    return 1 + calculateNumberOfNodes(node->left_child) + calculateNumberOfNodes(node->right_child);
}

int calculateMaxDepth(Node* node) {
    if (node == nullptr) return 0;
    if (node->is_leaf) return 1; // Leaf nodes count as depth 1

    int left_depth = calculateMaxDepth(node->left_child);
    int right_depth = calculateMaxDepth(node->right_child);

    return 1 + max(left_depth, right_depth);
}

// --- 5. Tree Printing (for visualization) ---

void printTree(Node* node, int indent = 0) {
    if (node == nullptr) return;

    for (int i = 0; i < indent; ++i) cout << "  ";

    if (node->is_leaf) {
        cout << "Leaf Node: Class = " << node->majority_class << " (Counts: ";
        bool first = true;
        for (const auto& pair : node->class_counts) {
            if (!first) cout << ", ";
            cout << pair.first << ":" << pair.second;
            first = false;
        }
        cout << ")\n";
    } else {
        cout << "Split on ";
        switch (node->feature) {
            case FeatureType::SepalLength: cout << "SepalLengthCm"; break;
            case FeatureType::SepalWidth:  cout << "SepalWidthCm"; break;
            case FeatureType::PetalLength: cout << "PetalLengthCm"; break;
            case FeatureType::PetalWidth:  cout << "PetalWidthCm"; break;
            default: cout << "UNKNOWN_FEATURE"; break;
        }
        cout << " <= " << node->threshold << "\n";

        cout << string(indent + 1, ' ') << "Left Child (<= " << node->threshold << "):\n";
        printTree(node->left_child, indent + 2);
        cout << string(indent + 1, ' ') << "Right Child (> " << node->threshold << "):\n";
        printTree(node->right_child, indent + 2);
    }
}

// --- Main function to demonstrate ---
int main(int argc, char* argv[]) {
    int maxDepth=-1;
    // Hardcoded a subset of your data for demonstration
    //load dataset from training_dataset80.csv separated by commas
    if(argc < 1){
        // cerr << "Usage: " << argv[0] << " <dataset_type>" << endl;
        // return 1;
        criterion = 0;
    }
    else if(string(argv[1])=="ig"){
        criterion = 0;
    } else if(string(argv[1])=="nwig"){
        criterion = 2;
    } else if(string(argv[1])=="igr"){
        criterion = 1;
    }
    else {criterion = 0;}


    if(argc>2){
        maxDepth = stoi(argv[2]);
    }

   if(maxDepth < 1) {
       cerr << "Invalid max depth. Using default value of 4." << endl;
       maxDepth = 4;
   }

    vector<DataRow> dataset;
    string line;
    ifstream file("training_dataset80.csv");
    if (!file.is_open()) {
        cerr << "Error: Could not open training_dataset80.csv" << endl;
        return 1;
    }
    while (getline(file, line))
    {
        stringstream ss(line);
        string token;

        // Read the attributes
        int id;
        double sepalLength, sepalWidth, petalLength, petalWidth;
        string species;

        getline(ss, token, ',');
        id = stoi(token);
        getline(ss, token, ',');
        sepalLength = stod(token);
        getline(ss, token, ',');
        sepalWidth = stod(token);
        getline(ss, token, ',');
        petalLength = stod(token);
        getline(ss, token, ',');
        petalWidth = stod(token);
        getline(ss, token, ',');
        species = token;

        // Create a DataRow object and add it to the dataset
        dataset.push_back({sepalLength, sepalWidth, petalLength, petalWidth, species});
    }

    file.close();
    cout<< "Dataset loaded with " << dataset.size() << " entries." << endl;
    

    // Define features
    vector<FeatureType> features_to_use = {
        FeatureType::SepalLength,
        FeatureType::SepalWidth,
        FeatureType::PetalLength,
        FeatureType::PetalWidth
    };

    // Build the tree
    // max_depth = -1 means no limit
    // min_samples_leaf = 5
    Node* root = buildDecisionTree(dataset, features_to_use, maxDepth, 5); // Example depth 4, min 5 samples

    cout << "--- Constructed Decision Tree (C++) ---\n";
    printTree(root);


    vector<DataRow> known_test_samples ;
    vector<DataRow> for_test_samples;

    //read from test_dataset20.csv
    ifstream test_file("testing_dataset20.csv");
    if (!test_file.is_open()) {
        cerr << "Error: Could not open test_dataset20.csv" << endl;
        return 1;
    }
    while (getline(test_file, line))
    {
        stringstream ss(line);
        string token;

        // Read the attributes
        double sepalLength, sepalWidth, petalLength, petalWidth;
        string species;

        getline(ss, token, ','); // Skip ID

        getline(ss, token, ',');
        sepalLength = stod(token);
        getline(ss, token, ',');
        sepalWidth = stod(token);
        getline(ss, token, ',');
        petalLength = stod(token);
        getline(ss, token, ',');
        petalWidth = stod(token);
        getline(ss, token, ',');
        species = token;

        // Create a DataRow object and add it to the test samples
        known_test_samples.push_back({sepalLength, sepalWidth, petalLength, petalWidth, species});
        for_test_samples.push_back({sepalLength, sepalWidth, petalLength, petalWidth, ""}); // Empty species for prediction
    }

    test_file.close();
    cout << "Test dataset loaded with " << known_test_samples.size() << " entries."
            << endl;

    int matched_count = 0;
    int unmatched_count = 0;
    cout << "\n--- Predictions Iris  (C++) ---\n";

   for(int i=0;i< for_test_samples.size(); i++) {
        string predicted_species = predict(root, for_test_samples[i]);
        for_test_samples[i].species = predicted_species; // Store prediction

        // Check against known test samples
        if (predicted_species == known_test_samples[i].species) {
            matched_count++;
        } else {
            unmatched_count++;
        }
        cout << "Sample " << i + 1 << ": Predicted = " << for_test_samples[i].species 
             << ", Actual = " << known_test_samples[i].species << endl;
    }

    cout << "\nTotal Matched: " << matched_count << ", Unmatched: " << unmatched_count 
         << endl;

    cout<<"percentage of matched samples: "
        << (static_cast<double>(matched_count) / for_test_samples.size()) * 100.0 
        << "%" << endl;

    //write the all detail predictions to a file
    ofstream predictions_file("predictions.csv");
    if (!predictions_file.is_open()) {
        cerr << "Error: Could not open predictions.csv for writing." << endl;
        return 1;
    }
    string tmp_ctr;
    // predictions_file << "SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,PredictedSpecies\n";
    if(criterion==0) tmp_ctr = "IG";
    else if(criterion==1) tmp_ctr = "IGR";
    else if(criterion==2) tmp_ctr = "NWIG";
    int derived_maxDepth = calculateMaxDepth(root)-1;
    //write criterion,accuracy,matched_count,unmatched_count
    predictions_file << "Criterion: " << tmp_ctr << endl
                     << "Accuracy: "<< (static_cast<double>(matched_count) / for_test_samples.size()) * 100.0 << "%" << endl
                     << "Depth: " << maxDepth<<","<<derived_maxDepth<< "\n"
                     <<"Node: " << calculateNumberOfNodes(root) << "\n";
    // --- Example Predictions ---
    // cout << "\n--- Predictions (C++) ---\n";
    // DataRow sample1 = {5.0, 3.5, 1.4, 0.2, ""}; // Known Setosa
    // DataRow sample2 = {6.0, 2.7, 4.5, 1.5, ""}; // Known Versicolor-ish
    // DataRow sample3 = {7.0, 3.2, 5.9, 2.1, ""}; // Known Virginica-ish

    // cout << "Sample 1 (5.0, 3.5, 1.4, 0.2) predicted as: " << predict(root, sample1) << endl;
    // cout << "Sample 2 (6.0, 2.7, 4.5, 1.5) predicted as: " << predict(root, sample2) << endl;
    // cout << "Sample 3 (7.0, 3.2, 5.9, 2.1) predicted as: " << predict(root, sample3) << endl;
    
    // Clean up memory
    delete root; 

    return 0;
}