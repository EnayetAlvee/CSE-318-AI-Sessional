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

// Represents a single row of data for adult dataset
struct DataRow {
    double age;
    string workclass;
    double fnlwgt;
    string education;
    double education_num;
    string marital_status;
    string occupation;
    string relationship;
    string race;
    string sex;
    double capital_gain;
    double capital_loss;
    double hours_per_week;
    string native_country;
    string income; // Target variable (<=50K or >50K)
};

// Represents a node in the Decision Tree
enum class FeatureType {
    Age,
    Workclass,
    Fnlwgt,
    Education,
    EducationNum,
    MaritalStatus,
    Occupation,
    Relationship,
    Race,
    Sex,
    CapitalGain,
    CapitalLoss,
    HoursPerWeek,
    NativeCountry,
    Leaf // Indicates a leaf node
};

// Helper function to convert string feature name to enum
FeatureType stringToFeatureType(const string& featureName) {
    if (featureName == "age") return FeatureType::Age;
    if (featureName == "workclass") return FeatureType::Workclass;
    if (featureName == "fnlwgt") return FeatureType::Fnlwgt;
    if (featureName == "education") return FeatureType::Education;
    if (featureName == "education_num") return FeatureType::EducationNum;
    if (featureName == "marital_status") return FeatureType::MaritalStatus;
    if (featureName == "occupation") return FeatureType::Occupation;
    if (featureName == "relationship") return FeatureType::Relationship;
    if (featureName == "race") return FeatureType::Race;
    if (featureName == "sex") return FeatureType::Sex;
    if (featureName == "capital_gain") return FeatureType::CapitalGain;
    if (featureName == "capital_loss") return FeatureType::CapitalLoss;
    if (featureName == "hours_per_week") return FeatureType::HoursPerWeek;
    if (featureName == "native_country") return FeatureType::NativeCountry;
    return FeatureType::Leaf;
}

// Helper to check if feature is categorical
bool isCategoricalFeature(FeatureType feature) {
    return feature == FeatureType::Workclass || 
           feature == FeatureType::Education || 
           feature == FeatureType::MaritalStatus ||
           feature == FeatureType::Occupation ||
           feature == FeatureType::Relationship ||
           feature == FeatureType::Race ||
           feature == FeatureType::Sex ||
           feature == FeatureType::NativeCountry;
}

// Helper to get numerical feature value from DataRow based on FeatureType
double getFeatureValue(const DataRow& row, FeatureType feature) {
    switch (feature) {
        case FeatureType::Age: return row.age;
        case FeatureType::Fnlwgt: return row.fnlwgt;
        case FeatureType::EducationNum: return row.education_num;
        case FeatureType::CapitalGain: return row.capital_gain;
        case FeatureType::CapitalLoss: return row.capital_loss;
        case FeatureType::HoursPerWeek: return row.hours_per_week;
        default: return 0.0; // For categorical features, this shouldn't be used
    }
}

// Helper to get categorical feature value
string getCategoricalValue(const DataRow& row, FeatureType feature) {
    switch (feature) {
        case FeatureType::Workclass: return row.workclass;
        case FeatureType::Education: return row.education;
        case FeatureType::MaritalStatus: return row.marital_status;
        case FeatureType::Occupation: return row.occupation;
        case FeatureType::Relationship: return row.relationship;
        case FeatureType::Race: return row.race;
        case FeatureType::Sex: return row.sex;
        case FeatureType::NativeCountry: return row.native_country;
        default: return "";
    }
}

struct Node {
    FeatureType feature;      // Feature used for splitting at this node
    double threshold;         // Split value for continuous features
    string categorical_value; // Split value for categorical features
    map<string, int> class_counts; // Class counts if it's a leaf node
    string majority_class; // Most frequent class in this node/leaf

    Node* left_child;         // Pointer to the left child
    Node* right_child;        // Pointer to the right child
    bool is_leaf;             // True if this is a leaf node
    bool is_categorical_split; // True if this is a categorical split

    // Constructor for internal node (continuous)
    Node(FeatureType f, double t) :
        feature(f), threshold(t), left_child(nullptr), right_child(nullptr), 
        is_leaf(false), is_categorical_split(false) {}

    // Constructor for internal node (categorical)
    Node(FeatureType f, string cat_val) :
        feature(f), threshold(0.0), categorical_value(cat_val), left_child(nullptr), 
        right_child(nullptr), is_leaf(false), is_categorical_split(true) {}

    // Constructor for leaf node
    Node(const vector<DataRow>& data) :
        feature(FeatureType::Leaf), threshold(0.0), left_child(nullptr), 
        right_child(nullptr), is_leaf(true), is_categorical_split(false) {
        
        // Calculate class counts for the leaf
        for (const auto& row : data) {
            class_counts[row.income]++;
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
        class_counts[row.income]++;
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

// Function to calculate Information Gain for categorical features
double calculateCategoricalInformationGain(const vector<DataRow>& data, FeatureType feature, const string& split_value) {
    double total_entropy = calculateEntropy(data);

    vector<DataRow> subset_equal, subset_not_equal;

    for (const auto& row : data) {
        if (getCategoricalValue(row, feature) == split_value) {
            subset_equal.push_back(row);
        } else {
            subset_not_equal.push_back(row);
        }
    }

    if (subset_equal.empty() || subset_not_equal.empty()) {
        return -1.0; // Invalid split
    }

    double entropy_equal = calculateEntropy(subset_equal);
    double entropy_not_equal = calculateEntropy(subset_not_equal);

    double weighted_entropy = (static_cast<double>(subset_equal.size()) / data.size()) * entropy_equal +
                              (static_cast<double>(subset_not_equal.size()) / data.size()) * entropy_not_equal;

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

double calculateCategoricalInformationGainRatio(const vector<DataRow>& data, FeatureType feature, const string& split_value) {
    double gain = calculateCategoricalInformationGain(data, feature, split_value);
    if (gain <= 0.0) return 0.0;

    double total_count = data.size();
    double subset_equal_count = 0.0;
    double subset_not_equal_count = 0.0;

    for (const auto& row : data) {
        if (getCategoricalValue(row, feature) == split_value) {
            subset_equal_count++;
        } else {
            subset_not_equal_count++;
        }
    }

    double intrinsic_value = 0.0;
    if (subset_equal_count > 0) {
        double prob_equal = subset_equal_count / total_count;
        intrinsic_value -= prob_equal * log2(prob_equal);
    }
    if (subset_not_equal_count > 0) {
        double prob_not_equal = subset_not_equal_count / total_count;
        intrinsic_value -= prob_not_equal * log2(prob_not_equal);
    }

    return intrinsic_value > 0 ? gain / intrinsic_value : 0.0;
}

// Calculate NWIG for continuous features
double calculateNWIG(const vector<DataRow>& data, FeatureType feature, double split_value) {
    double gain = calculateInformationGain(data, feature, split_value);
    if (gain <= 0.0) return 0.0;

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

// Calculate NWIG for categorical features
double calculateCategoricalNWIG(const vector<DataRow>& data, FeatureType feature, const string& split_value) {
    double gain = calculateCategoricalInformationGain(data, feature, split_value);
    if (gain <= 0.0) return 0.0;

    // Count distinct values for the categorical feature
    set<string> unique_values_set;
    for (const auto& row : data) {
        unique_values_set.insert(getCategoricalValue(row, feature));
    }
    double k = static_cast<double>(unique_values_set.size());
    
    double dataset_size = static_cast<double>(data.size());
    
    if (k <= 1 || dataset_size <= 1) return 0.0;
    
    double normalization_factor = gain / log2(k + 1.0);
    double penalty_factor = 1.0 - (k - 1.0) / dataset_size;
    
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
        double current_gain;
        
        if(criterion == 0) {
            current_gain = calculateInformationGain(data, feature, split_value);
        } else if(criterion == 2) {
            current_gain = calculateNWIG(data, feature, split_value);
        } else if(criterion == 1) {
            current_gain = calculateInformationGainRatio(data, feature, split_value);
        } else {
            cerr << "Unknown criterion: " << criterion << endl;
            return {0.0, -1.0};
        }
        
        if (current_gain > best_gain) {
            best_gain = current_gain;
            best_split_value = split_value;
        }
    }
    return {best_split_value, best_gain};
}

// Function to find the best split for categorical features
pair<string, double> findBestCategoricalSplit(const vector<DataRow>& data, FeatureType feature) {
    set<string> unique_values_set;
    for (const auto& row : data) {
        unique_values_set.insert(getCategoricalValue(row, feature));
    }

    vector<string> unique_values(unique_values_set.begin(), unique_values_set.end());
    
    if (unique_values.size() < 2) {
        return {"", -1.0}; // Cannot split if less than 2 unique values
    }

    double best_gain = -1.0;
    string best_split_value = "";

    for (const auto& split_value : unique_values) {
        double current_gain;
        
        if(criterion == 0) {
            current_gain = calculateCategoricalInformationGain(data, feature, split_value);
        } else if(criterion == 2) {
            current_gain = calculateCategoricalNWIG(data, feature, split_value);
        } else if(criterion == 1) {
            current_gain = calculateCategoricalInformationGainRatio(data, feature, split_value);
        } else {
            cerr << "Unknown criterion: " << criterion << endl;
            return {"", -1.0};
        }
        
        if (current_gain > best_gain) {
            best_gain = current_gain;
            best_split_value = split_value;
        }
    }
    return {best_split_value, best_gain};
}

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
    FeatureType best_feature = FeatureType::Leaf;
    double best_threshold = 0.0;
    string best_categorical_value = "";
    bool best_is_categorical = false;

    // Find the best split among available features
    for (FeatureType feature : features_available) {
        if (isCategoricalFeature(feature)) {
            // Handle categorical feature
            pair<string, double> split_info = findBestCategoricalSplit(data, feature);
            string current_split_value = split_info.first;
            double current_gain = split_info.second;

            if (current_gain > best_gain) {
                best_gain = current_gain;
                best_feature = feature;
                best_categorical_value = current_split_value;
                best_is_categorical = true;
            }
        } else {
            // Handle continuous feature
            pair<double, double> split_info = findBestContinuousSplit(data, feature);
            double current_threshold = split_info.first;
            double current_gain = split_info.second;

            if (current_gain > best_gain) {
                best_gain = current_gain;
                best_feature = feature;
                best_threshold = current_threshold;
                best_is_categorical = false;
            }
        }
    }

    // Base Case 4: No significant gain from any split
    if (best_gain <= 0.0) {
        return new Node(data); // Create a leaf node
    }

    // Create a new internal node
    Node* node;
    if (best_is_categorical) {
        node = new Node(best_feature, best_categorical_value);
    } else {
        node = new Node(best_feature, best_threshold);
    }

    // Split data for children
    vector<DataRow> left_data, right_data;
    for (const auto& row : data) {
        if (best_is_categorical) {
            if (getCategoricalValue(row, best_feature) == best_categorical_value) {
                left_data.push_back(row);
            } else {
                right_data.push_back(row);
            }
        } else {
            if (getFeatureValue(row, best_feature) <= best_threshold) {
                left_data.push_back(row);
            } else {
                right_data.push_back(row);
            }
        }
    }

    // Handle empty splits
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

    if (node->is_categorical_split) {
        string feature_value = getCategoricalValue(sample, node->feature);
        if (feature_value == node->categorical_value) {
            return predict(node->left_child, sample);
        } else {
            return predict(node->right_child, sample);
        }
    } else {
        double feature_value = getFeatureValue(sample, node->feature);
        if (feature_value <= node->threshold) {
            return predict(node->left_child, sample);
        } else {
            return predict(node->right_child, sample);
        }
    }
}

int calculateNumberOfNodes(Node* node) {
    if (node == nullptr) return 0;
    return 1 + calculateNumberOfNodes(node->left_child) + calculateNumberOfNodes(node->right_child);
}

int calculateMaxDepth(Node* node) {
    if (node == nullptr) return 0;
    if (node->is_leaf) return 1;

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
            case FeatureType::Age: cout << "Age"; break;
            case FeatureType::Workclass: cout << "Workclass"; break;
            case FeatureType::Fnlwgt: cout << "Fnlwgt"; break;
            case FeatureType::Education: cout << "Education"; break;
            case FeatureType::EducationNum: cout << "Education-num"; break;
            case FeatureType::MaritalStatus: cout << "Marital-status"; break;
            case FeatureType::Occupation: cout << "Occupation"; break;
            case FeatureType::Relationship: cout << "Relationship"; break;
            case FeatureType::Race: cout << "Race"; break;
            case FeatureType::Sex: cout << "Sex"; break;
            case FeatureType::CapitalGain: cout << "Capital-gain"; break;
            case FeatureType::CapitalLoss: cout << "Capital-loss"; break;
            case FeatureType::HoursPerWeek: cout << "Hours-per-week"; break;
            case FeatureType::NativeCountry: cout << "Native-country"; break;
            default: cout << "UNKNOWN_FEATURE"; break;
        }
        
        if (node->is_categorical_split) {
            cout << " == " << node->categorical_value << "\n";
            cout << string(indent + 1, ' ') << "Left Child (== " << node->categorical_value << "):\n";
            printTree(node->left_child, indent + 2);
            cout << string(indent + 1, ' ') << "Right Child (!= " << node->categorical_value << "):\n";
            printTree(node->right_child, indent + 2);
        } else {
            cout << " <= " << node->threshold << "\n";
            cout << string(indent + 1, ' ') << "Left Child (<= " << node->threshold << "):\n";
            printTree(node->left_child, indent + 2);
            cout << string(indent + 1, ' ') << "Right Child (> " << node->threshold << "):\n";
            printTree(node->right_child, indent + 2);
        }
    }
}

// Function to trim whitespace from string
string trim(const string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Function to load raw data with missing values marked as "MISSING"
vector<DataRow> loadRawData(const string& filename) {
    vector<DataRow> dataset;
    ifstream file(filename);
    string line;
    
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return dataset;
    }
    
    while (getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines
        
        stringstream ss(line);
        string token;
        DataRow row;
        
        try {
            // Age
            getline(ss, token, ',');
            string age_str = trim(token);
            if (age_str == "?" || age_str.empty()) {
                row.age = -999.0; // Special marker for missing numerical values
            } else {
                row.age = stod(age_str);
            }
            
            // Workclass
            getline(ss, token, ',');
            row.workclass = trim(token);
            if (row.workclass == "?" || row.workclass.empty()) {
                row.workclass = "MISSING";
            }
            
            // Fnlwgt
            getline(ss, token, ',');
            string fnlwgt_str = trim(token);
            if (fnlwgt_str == "?" || fnlwgt_str.empty()) {
                row.fnlwgt = -999.0;
            } else {
                row.fnlwgt = stod(fnlwgt_str);
            }
            
            // Education
            getline(ss, token, ',');
            row.education = trim(token);
            if (row.education == "?" || row.education.empty()) {
                row.education = "MISSING";
            }
            
            // Education-num
            getline(ss, token, ',');
            string edu_num_str = trim(token);
            if (edu_num_str == "?" || edu_num_str.empty()) {
                row.education_num = -999.0;
            } else {
                row.education_num = stod(edu_num_str);
            }
            
            // Marital-status
            getline(ss, token, ',');
            row.marital_status = trim(token);
            if (row.marital_status == "?" || row.marital_status.empty()) {
                row.marital_status = "MISSING";
            }
            
            // Occupation
            getline(ss, token, ',');
            row.occupation = trim(token);
            if (row.occupation == "?" || row.occupation.empty()) {
                row.occupation = "MISSING";
            }
            
            // Relationship
            getline(ss, token, ',');
            row.relationship = trim(token);
            if (row.relationship == "?" || row.relationship.empty()) {
                row.relationship = "MISSING";
            }
            
            // Race
            getline(ss, token, ',');
            row.race = trim(token);
            if (row.race == "?" || row.race.empty()) {
                row.race = "MISSING";
            }
            
            // Sex
            getline(ss, token, ',');
            row.sex = trim(token);
            if (row.sex == "?" || row.sex.empty()) {
                row.sex = "MISSING";
            }
            
            // Capital-gain
            getline(ss, token, ',');
            string cap_gain_str = trim(token);
            if (cap_gain_str == "?" || cap_gain_str.empty()) {
                row.capital_gain = -999.0;
            } else {
                row.capital_gain = stod(cap_gain_str);
            }
            
            // Capital-loss
            getline(ss, token, ',');
            string cap_loss_str = trim(token);
            if (cap_loss_str == "?" || cap_loss_str.empty()) {
                row.capital_loss = -999.0;
            } else {
                row.capital_loss = stod(cap_loss_str);
            }
            
            // Hours-per-week
            getline(ss, token, ',');
            string hours_str = trim(token);
            if (hours_str == "?" || hours_str.empty()) {
                row.hours_per_week = -999.0;
            } else {
                row.hours_per_week = stod(hours_str);
            }
            
            // Native-country
            getline(ss, token, ',');
            row.native_country = trim(token);
            if (row.native_country == "?" || row.native_country.empty()) {
                row.native_country = "MISSING";
            }
            
            // Income (target)
            getline(ss, token, ',');
            row.income = trim(token);
            
            dataset.push_back(row);
            
        } catch (const exception& e) {
            cerr << "Error parsing line: " << line << endl;
            cerr << "Error: " << e.what() << endl;
            continue; // Skip this line and continue
        }
    }
    
    file.close();
    return dataset;
}

// Function to calculate most frequent value for categorical attributes
string calculateMostFrequentCategorical(const vector<DataRow>& data, const string& attribute) {
    map<string, int> frequency;
    
    for (const auto& row : data) {
        string value;
        if (attribute == "workclass") value = row.workclass;
        else if (attribute == "education") value = row.education;
        else if (attribute == "marital_status") value = row.marital_status;
        else if (attribute == "occupation") value = row.occupation;
        else if (attribute == "relationship") value = row.relationship;
        else if (attribute == "race") value = row.race;
        else if (attribute == "sex") value = row.sex;
        else if (attribute == "native_country") value = row.native_country;
        
        if (value != "MISSING") {
            frequency[value]++;
        }
    }
    
    string mostFrequent = "Unknown";
    int maxCount = 0;
    for (const auto& pair : frequency) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostFrequent = pair.first;
        }
    }
    
    return mostFrequent;
}

// Function to calculate most frequent value for numerical attributes
double calculateMostFrequentNumerical(const vector<DataRow>& data, const string& attribute) {
    map<double, int> frequency;
    
    for (const auto& row : data) {
        double value;
        if (attribute == "age") value = row.age;
        else if (attribute == "fnlwgt") value = row.fnlwgt;
        else if (attribute == "education_num") value = row.education_num;
        else if (attribute == "capital_gain") value = row.capital_gain;
        else if (attribute == "capital_loss") value = row.capital_loss;
        else if (attribute == "hours_per_week") value = row.hours_per_week;
        else continue;
        
        if (value != -999.0) {
            frequency[value]++;
        }
    }
    
    double mostFrequent = 0.0;
    int maxCount = 0;
    for (const auto& pair : frequency) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostFrequent = pair.first;
        }
    }
    
    return mostFrequent;
}

// Function to replace missing values with most frequent values
void replaceMissingValues(vector<DataRow>& data) {
    cout << "Calculating most frequent values for missing data imputation..." << endl;
    
    // Calculate most frequent values for categorical attributes
    string workclass_mode = calculateMostFrequentCategorical(data, "workclass");
    string education_mode = calculateMostFrequentCategorical(data, "education");
    string marital_status_mode = calculateMostFrequentCategorical(data, "marital_status");
    string occupation_mode = calculateMostFrequentCategorical(data, "occupation");
    string relationship_mode = calculateMostFrequentCategorical(data, "relationship");
    string race_mode = calculateMostFrequentCategorical(data, "race");
    string sex_mode = calculateMostFrequentCategorical(data, "sex");
    string native_country_mode = calculateMostFrequentCategorical(data, "native_country");
    
    // Calculate most frequent values for numerical attributes
    double age_mode = calculateMostFrequentNumerical(data, "age");
    double fnlwgt_mode = calculateMostFrequentNumerical(data, "fnlwgt");
    double education_num_mode = calculateMostFrequentNumerical(data, "education_num");
    double capital_gain_mode = calculateMostFrequentNumerical(data, "capital_gain");
    double capital_loss_mode = calculateMostFrequentNumerical(data, "capital_loss");
    double hours_per_week_mode = calculateMostFrequentNumerical(data, "hours_per_week");
    
    // Print the calculated most frequent values
    cout << "Most frequent values for imputation:" << endl;
    cout << "  Workclass: " << workclass_mode << endl;
    cout << "  Education: " << education_mode << endl;
    cout << "  Marital-status: " << marital_status_mode << endl;
    cout << "  Occupation: " << occupation_mode << endl;
    cout << "  Relationship: " << relationship_mode << endl;
    cout << "  Race: " << race_mode << endl;
    cout << "  Sex: " << sex_mode << endl;
    cout << "  Native-country: " << native_country_mode << endl;
    cout << "  Age: " << age_mode << endl;
    cout << "  Fnlwgt: " << fnlwgt_mode << endl;
    cout << "  Education-num: " << education_num_mode << endl;
    cout << "  Capital-gain: " << capital_gain_mode << endl;
    cout << "  Capital-loss: " << capital_loss_mode << endl;
    cout << "  Hours-per-week: " << hours_per_week_mode << endl;
    cout << endl;
    
    // Replace missing values in the dataset
    int missing_count = 0;
    for (auto& row : data) {
        // Replace categorical missing values
        if (row.workclass == "MISSING") {
            row.workclass = workclass_mode;
            missing_count++;
        }
        if (row.education == "MISSING") {
            row.education = education_mode;
            missing_count++;
        }
        if (row.marital_status == "MISSING") {
            row.marital_status = marital_status_mode;
            missing_count++;
        }
        if (row.occupation == "MISSING") {
            row.occupation = occupation_mode;
            missing_count++;
        }
        if (row.relationship == "MISSING") {
            row.relationship = relationship_mode;
            missing_count++;
        }
        if (row.race == "MISSING") {
            row.race = race_mode;
            missing_count++;
        }
        if (row.sex == "MISSING") {
            row.sex = sex_mode;
            missing_count++;
        }
        if (row.native_country == "MISSING") {
            row.native_country = native_country_mode;
            missing_count++;
        }
        
        // Replace numerical missing values
        if (row.age == -999.0) {
            row.age = age_mode;
            missing_count++;
        }
        if (row.fnlwgt == -999.0) {
            row.fnlwgt = fnlwgt_mode;
            missing_count++;
        }
        if (row.education_num == -999.0) {
            row.education_num = education_num_mode;
            missing_count++;
        }
        if (row.capital_gain == -999.0) {
            row.capital_gain = capital_gain_mode;
            missing_count++;
        }
        if (row.capital_loss == -999.0) {
            row.capital_loss = capital_loss_mode;
            missing_count++;
        }
        if (row.hours_per_week == -999.0) {
            row.hours_per_week = hours_per_week_mode;
            missing_count++;
        }
    }
    
    cout << "Total missing values replaced: " << missing_count << endl;
}

// --- Main function ---
int main(int argc, char* argv[]) {
    int maxDepth = -1;
    
    if(argc < 1){
        criterion = 0;
    } else if(string(argv[1]) == "ig"){
        criterion = 0;
    } else if(string(argv[1]) == "nwig"){
        criterion = 2;
    } else if(string(argv[1]) == "igr"){
        criterion = 1;
    } else {
        criterion = 0;
    }

    if(argc > 2){
        maxDepth = stoi(argv[2]);
    }

    if(maxDepth < 1) {
        cerr << "Invalid max depth. Using default value of 4." << endl;
        maxDepth = 4;
    }

    // Step 1: Load training data with missing values marked as "MISSING" or -999.0
    cout << "Step 1: Loading training data with missing values marked..." << endl;
    vector<DataRow> dataset = loadRawData("training_dataset80.csv");
    cout << "Raw dataset loaded with " << dataset.size() << " entries." << endl;
    
    // Step 2: Calculate most frequent values and replace missing values
    cout << "Step 2: Replacing missing values with most frequent values..." << endl;
    replaceMissingValues(dataset);
    cout << "Dataset after imputation: " << dataset.size() << " entries." << endl;
    cout << endl;

    // Define features to use (only numerical features for simplicity, following task1 style)
    vector<FeatureType> features_to_use = {
        FeatureType::Age,
        FeatureType::EducationNum,
        FeatureType::CapitalGain,
        FeatureType::CapitalLoss,
        FeatureType::HoursPerWeek
    };

    // Build the tree
    Node* root = buildDecisionTree(dataset, features_to_use, maxDepth, 5);

    cout << "--- Constructed Decision Tree (Adult Dataset) ---\n";
    printTree(root);

    // Test on testing dataset - use same imputation approach
    vector<DataRow> test_dataset = loadRawData("testing_dataset20.csv");
    cout << "Raw test dataset loaded with " << test_dataset.size() << " entries." << endl;
    
    // For test data, we need to apply the same imputation strategy as training data
    // We'll use the training data to calculate imputation values and apply to test data
    cout << "Applying same imputation strategy to test data..." << endl;
    replaceMissingValues(test_dataset);

    vector<DataRow> known_test_samples = test_dataset;
    vector<DataRow> for_test_samples;

    // Create copies for prediction (without target)
    for (const auto& row : test_dataset) {
        DataRow test_row = row;
        test_row.income = "";
        for_test_samples.push_back(test_row);
    }

    int matched_count = 0;
    int unmatched_count = 0;
    cout << "\n--- Predictions Adult Dataset ---\n";

    for(int i = 0; i < for_test_samples.size(); i++) {
        string predicted_income = predict(root, for_test_samples[i]);
        for_test_samples[i].income = predicted_income;

        if (predicted_income == known_test_samples[i].income) {
            matched_count++;
        } else {
            unmatched_count++;
        }
        
        if (i < 10) { // Show first 10 predictions
            cout << "Sample " << i + 1 << ": Predicted = " << for_test_samples[i].income 
                 << ", Actual = " << known_test_samples[i].income << endl;
        }
    }

    cout << "\nTotal Matched: " << matched_count << ", Unmatched: " << unmatched_count << endl;
    cout << "Accuracy: " << (static_cast<double>(matched_count) / for_test_samples.size()) * 100.0 << "%" << endl;

    // Write predictions to file
    ofstream predictions_file("adult_predictions.txt");
    if (!predictions_file.is_open()) {
        cerr << "Error: Could not open adult_predictions.csv for writing." << endl;
        return 1;
    }

    string tmp_ctr;
    if(criterion == 0) tmp_ctr = "IG";
    else if(criterion == 1) tmp_ctr = "IGR";
    else if(criterion == 2) tmp_ctr = "NWIG";
    
    int derived_maxDepth = calculateMaxDepth(root) - 1;
    
    predictions_file << "Criterion: " << tmp_ctr << endl
                     << "Accuracy: " << (static_cast<double>(matched_count) / for_test_samples.size()) * 100.0 << "%" << endl
                     << "Depth: " << maxDepth << "," << derived_maxDepth << "\n"
                     << "Nodes: " << calculateNumberOfNodes(root) << "\n";

    predictions_file.close();

    // Clean up memory
    delete root;

    return 0;
}
