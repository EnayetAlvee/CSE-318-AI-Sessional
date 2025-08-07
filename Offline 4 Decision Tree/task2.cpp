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

const string MISSING_VALUE = "MISSING_VALUE";

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
    string income; // Target variable
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

// Helper to get feature value from DataRow based on FeatureType
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

// --- 2. Missing Value Handling ---

// Function to find most common value in a categorical column
string findMostCommonCategorical(const vector<DataRow>& data, FeatureType feature) {
    map<string, int> value_counts;
    
    for (const auto& row : data) {
        string value = getCategoricalValue(row, feature);
        if (value != MISSING_VALUE) {
            value_counts[value]++;
        }
    }
    
    string most_common = "";
    int max_count = 0;
    for (const auto& pair : value_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    
    return most_common;
}

// Function to find most common value in a numerical column
double findMostCommonNumerical(const vector<DataRow>& data, FeatureType feature) {
    map<double, int> value_counts;
    
    for (const auto& row : data) {
        double value = getFeatureValue(row, feature);
        if (value != -999.0) { // Assuming -999 represents missing numerical values
            value_counts[value]++;
        }
    }
    
    double most_common = 0.0;
    int max_count = 0;
    for (const auto& pair : value_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    
    return most_common;
}

// Function to replace missing values with most common values
void replaceMissingValues(vector<DataRow>& data) {
    // Find most common values for each feature
    vector<FeatureType> categorical_features = {
        FeatureType::Workclass, FeatureType::Education, FeatureType::MaritalStatus,
        FeatureType::Occupation, FeatureType::Relationship, FeatureType::Race,
        FeatureType::Sex, FeatureType::NativeCountry
    };
    
    vector<FeatureType> numerical_features = {
        FeatureType::Age, FeatureType::Fnlwgt, FeatureType::EducationNum,
        FeatureType::CapitalGain, FeatureType::CapitalLoss, FeatureType::HoursPerWeek
    };
    
    // Replace categorical missing values
    for (FeatureType feature : categorical_features) {
        string most_common = findMostCommonCategorical(data, feature);
        
        for (auto& row : data) {
            switch (feature) {
                case FeatureType::Workclass:
                    if (row.workclass == MISSING_VALUE) row.workclass = most_common;
                    break;
                case FeatureType::Education:
                    if (row.education == MISSING_VALUE) row.education = most_common;
                    break;
                case FeatureType::MaritalStatus:
                    if (row.marital_status == MISSING_VALUE) row.marital_status = most_common;
                    break;
                case FeatureType::Occupation:
                    if (row.occupation == MISSING_VALUE) row.occupation = most_common;
                    break;
                case FeatureType::Relationship:
                    if (row.relationship == MISSING_VALUE) row.relationship = most_common;
                    break;
                case FeatureType::Race:
                    if (row.race == MISSING_VALUE) row.race = most_common;
                    break;
                case FeatureType::Sex:
                    if (row.sex == MISSING_VALUE) row.sex = most_common;
                    break;
                case FeatureType::NativeCountry:
                    if (row.native_country == MISSING_VALUE) row.native_country = most_common;
                    break;
            }
        }
    }
    
    // Replace numerical missing values (if any were marked as -999)
    for (FeatureType feature : numerical_features) {
        double most_common = findMostCommonNumerical(data, feature);
        
        for (auto& row : data) {
            switch (feature) {
                case FeatureType::Age:
                    if (row.age == -999.0) row.age = most_common;
                    break;
                case FeatureType::Fnlwgt:
                    if (row.fnlwgt == -999.0) row.fnlwgt = most_common;
                    break;
                case FeatureType::EducationNum:
                    if (row.education_num == -999.0) row.education_num = most_common;
                    break;
                case FeatureType::CapitalGain:
                    if (row.capital_gain == -999.0) row.capital_gain = most_common;
                    break;
                case FeatureType::CapitalLoss:
                    if (row.capital_loss == -999.0) row.capital_loss = most_common;
                    break;
                case FeatureType::HoursPerWeek:
                    if (row.hours_per_week == -999.0) row.hours_per_week = most_common;
                    break;
            }
        }
    }
}

// --- 3. Core Calculation Functions ---

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
double calculateCategoricalInformationGain(const vector<DataRow>& data, FeatureType feature, const string& category_value) {
    double total_entropy = calculateEntropy(data);

    vector<DataRow> subset_match, subset_other;

    for (const auto& row : data) {
        if (getCategoricalValue(row, feature) == category_value) {
            subset_match.push_back(row);
        } else {
            subset_other.push_back(row);
        }
    }

    if (subset_match.empty() || subset_other.empty()) {
        return -1.0; // Invalid split
    }

    double entropy_match = calculateEntropy(subset_match);
    double entropy_other = calculateEntropy(subset_other);

    double weighted_entropy = (static_cast<double>(subset_match.size()) / data.size()) * entropy_match +
                              (static_cast<double>(subset_other.size()) / data.size()) * entropy_other;

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

double calculateCategoricalInformationGainRatio(const vector<DataRow>& data, FeatureType feature, const string& category_value) {
    double gain = calculateCategoricalInformationGain(data, feature, category_value);
    if (gain <= 0.0) return 0.0;

    double total_count = data.size();
    double subset_match_count = 0.0;
    double subset_other_count = 0.0;

    for (const auto& row : data) {
        if (getCategoricalValue(row, feature) == category_value) {
            subset_match_count++;
        } else {
            subset_other_count++;
        }
    }

    double intrinsic_value = 0.0;
    if (subset_match_count > 0) {
        double prob_match = subset_match_count / total_count;
        intrinsic_value -= prob_match * log2(prob_match);
    }
    if (subset_other_count > 0) {
        double prob_other = subset_other_count / total_count;
        intrinsic_value -= prob_other * log2(prob_other);
    }

    return intrinsic_value > 0 ? gain / intrinsic_value : 0.0;
}

// Calculate NWIG (Normalized Weighted Information Gain)
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

double calculateCategoricalNWIG(const vector<DataRow>& data, FeatureType feature, const string& category_value) {
    double gain = calculateCategoricalInformationGain(data, feature, category_value);
    if (gain <= 0.0) return 0.0;

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

// Function to find the best split for categorical attributes
pair<string, double> findBestCategoricalSplit(const vector<DataRow>& data, FeatureType feature) {
    set<string> unique_values_set;
    for (const auto& row : data) {
        unique_values_set.insert(getCategoricalValue(row, feature));
    }

    if (unique_values_set.size() < 2) {
        return {"", -1.0};
    }

    double best_gain = -1.0;
    string best_category = "";

    for (const string& category : unique_values_set) {
        double current_gain;
        
        if(criterion == 0) {
            current_gain = calculateCategoricalInformationGain(data, feature, category);
        } else if(criterion == 2) {
            current_gain = calculateCategoricalNWIG(data, feature, category);
        } else if(criterion == 1) {
            current_gain = calculateCategoricalInformationGainRatio(data, feature, category);
        } else {
            cerr << "Unknown criterion: " << criterion << endl;
            return {"", -1.0};
        }
        
        if (current_gain > best_gain) {
            best_gain = current_gain;
            best_category = category;
        }
    }
    return {best_category, best_gain};
}

// --- 4. Decision Tree Building ---

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
    string best_category = "";
    bool is_categorical = false;

    // Find the best split among available features
    for (FeatureType feature : features_available) {
        if (isCategoricalFeature(feature)) {
            pair<string, double> split_info = findBestCategoricalSplit(data, feature);
            string current_category = split_info.first;
            double current_gain = split_info.second;

            if (current_gain > best_gain) {
                best_gain = current_gain;
                best_feature = feature;
                best_category = current_category;
                is_categorical = true;
            }
        } else {
            pair<double, double> split_info = findBestContinuousSplit(data, feature);
            double current_threshold = split_info.first;
            double current_gain = split_info.second;

            if (current_gain > best_gain) {
                best_gain = current_gain;
                best_feature = feature;
                best_threshold = current_threshold;
                is_categorical = false;
            }
        }
    }

    // Base Case 4: No significant gain from any split
    if (best_gain <= 0.0) {
        return new Node(data); // Create a leaf node
    }

    // Create a new internal node
    Node* node;
    if (is_categorical) {
        node = new Node(best_feature, best_category);
    } else {
        node = new Node(best_feature, best_threshold);
    }

    // Split data for children
    vector<DataRow> left_data, right_data;
    
    if (is_categorical) {
        for (const auto& row : data) {
            if (getCategoricalValue(row, best_feature) == best_category) {
                left_data.push_back(row);
            } else {
                right_data.push_back(row);
            }
        }
    } else {
        for (const auto& row : data) {
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

// --- 5. Prediction Function ---

string predict(Node* node, const DataRow& sample) {
    if (node->is_leaf) {
        return node->majority_class;
    }

    if (node->is_categorical_split) {
        if (getCategoricalValue(sample, node->feature) == node->categorical_value) {
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

// --- 6. Tree Printing ---

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
            case FeatureType::Age: cout << "age"; break;
            case FeatureType::Workclass: cout << "workclass"; break;
            case FeatureType::Fnlwgt: cout << "fnlwgt"; break;
            case FeatureType::Education: cout << "education"; break;
            case FeatureType::EducationNum: cout << "education_num"; break;
            case FeatureType::MaritalStatus: cout << "marital_status"; break;
            case FeatureType::Occupation: cout << "occupation"; break;
            case FeatureType::Relationship: cout << "relationship"; break;
            case FeatureType::Race: cout << "race"; break;
            case FeatureType::Sex: cout << "sex"; break;
            case FeatureType::CapitalGain: cout << "capital_gain"; break;
            case FeatureType::CapitalLoss: cout << "capital_loss"; break;
            case FeatureType::HoursPerWeek: cout << "hours_per_week"; break;
            case FeatureType::NativeCountry: cout << "native_country"; break;
            default: cout << "UNKNOWN_FEATURE"; break;
        }
        
        if (node->is_categorical_split) {
            cout << " == " << node->categorical_value << "\n";
        } else {
            cout << " <= " << node->threshold << "\n";
        }

        cout << string(indent + 1, ' ') << "Left Child:\n";
        printTree(node->left_child, indent + 2);
        cout << string(indent + 1, ' ') << "Right Child:\n";
        printTree(node->right_child, indent + 2);
    }
}

// --- 7. Data Loading Functions ---

DataRow parseDataRow(const string& line) {
    DataRow row;
    stringstream ss(line);
    string token;
    
    // Parse each field
    getline(ss, token, ','); // age
    row.age = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // workclass
    row.workclass = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // fnlwgt
    row.fnlwgt = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // education
    row.education = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // education_num
    row.education_num = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // marital_status
    row.marital_status = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // occupation
    row.occupation = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // relationship
    row.relationship = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // race
    row.race = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // sex
    row.sex = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // capital_gain
    row.capital_gain = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // capital_loss
    row.capital_loss = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // hours_per_week
    row.hours_per_week = (token == "?") ? -999.0 : stod(token);
    
    getline(ss, token, ','); // native_country
    row.native_country = (token == "?") ? MISSING_VALUE : token;
    
    getline(ss, token, ','); // income (target variable)
    row.income = token;
    
    return row;
}

// --- Main function ---
int main(int argc, char* argv[]) {
    int maxDepth = -1;
    
    if(argc < 1){
        criterion = 0;
    }
    else if(string(argv[1]) == "ig"){
        criterion = 0;
    } else if(string(argv[1]) == "nwig"){
        criterion = 2;
    } else if(string(argv[1]) == "igr"){
        criterion = 1;
    }
    else {
        criterion = 0;
    }

    if(argc > 2){
        maxDepth = stoi(argv[2]);
    }

    if(maxDepth < 1) {
        cerr << "Invalid max depth. Using default value of 4." << endl;
        maxDepth = 4;
    }

    // Load training dataset
    vector<DataRow> dataset;
    string line;
    ifstream file("adult.data");
    if (!file.is_open()) {
        cerr << "Error: Could not open adult.data" << endl;
        return 1;
    }
    
    while (getline(file, line)) {
        if (!line.empty()) {
            DataRow row = parseDataRow(line);
            dataset.push_back(row);
        }
    }
    file.close();
    
    cout << "Dataset loaded with " << dataset.size() << " entries." << endl;
    
    // Replace missing values with most common values
    cout << "Replacing missing values..." << endl;
    replaceMissingValues(dataset);
    cout << "Missing values replaced." << endl;

    // Define features to use
    vector<FeatureType> features_to_use = {
        FeatureType::Age,
        FeatureType::Workclass,
        FeatureType::Fnlwgt,
        FeatureType::Education,
        FeatureType::EducationNum,
        FeatureType::MaritalStatus,
        FeatureType::Occupation,
        FeatureType::Relationship,
        FeatureType::Race,
        FeatureType::Sex,
        FeatureType::CapitalGain,
        FeatureType::CapitalLoss,
        FeatureType::HoursPerWeek,
        FeatureType::NativeCountry
    };

    // Build the tree
    cout << "Building decision tree..." << endl;
    Node* root = buildDecisionTree(dataset, features_to_use, maxDepth, 5);

    cout << "--- Constructed Decision Tree (Adult Dataset) ---\n";
    printTree(root);

    // Load test dataset
    vector<DataRow> known_test_samples;
    vector<DataRow> for_test_samples;

    ifstream test_file("adult.test");
    if (!test_file.is_open()) {
        cerr << "Error: Could not open adult.test" << endl;
        return 1;
    }
    
    while (getline(test_file, line)) {
        if (!line.empty()) {
            DataRow row = parseDataRow(line);
            known_test_samples.push_back(row);
            
            // Create test sample without target for prediction
            DataRow test_row = row;
            test_row.income = ""; // Empty income for prediction
            for_test_samples.push_back(test_row);
        }
    }
    test_file.close();
    
    cout << "Test dataset loaded with " << known_test_samples.size() << " entries." << endl;
    
    // Replace missing values in test data
    replaceMissingValues(for_test_samples);

    // Make predictions
    int matched_count = 0;
    int unmatched_count = 0;
    cout << "\n--- Predictions Adult Dataset ---\n";

    for(size_t i = 0; i < for_test_samples.size(); i++) {
        string predicted_income = predict(root, for_test_samples[i]);
        for_test_samples[i].income = predicted_income;

        // Check against known test samples
        if (predicted_income == known_test_samples[i].income) {
            matched_count++;
        } else {
            unmatched_count++;
        }
        
        if (i < 20) { // Print first 20 predictions
            cout << "Sample " << i + 1 << ": Predicted = " << for_test_samples[i].income 
                 << ", Actual = " << known_test_samples[i].income << endl;
        }
    }

    cout << "\nTotal Matched: " << matched_count << ", Unmatched: " << unmatched_count << endl;
    cout << "Accuracy: " << (static_cast<double>(matched_count) / for_test_samples.size()) * 100.0 << "%" << endl;

    // Write predictions to file
    ofstream predictions_file("adult_predictions.csv");
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