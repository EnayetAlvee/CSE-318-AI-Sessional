#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>

using namespace std;

void readCSV(const string& filename, vector<vector<string>>& data) {
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<string> row;
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }
        data.push_back(row);
    }
}

void writeCSV(const string& filename, const vector<vector<string>>& data) {
    ofstream file(filename);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << endl;
    }
}

void writeAttributes(const string& filename, const vector<string>& attributes) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    for (size_t i = 0; i < attributes.size(); ++i) {
        file << attributes[i];
        if (i < attributes.size() - 1) file << ",";
    }
    file << endl;

    file.close();
}


int main(int argc, char* argv[]) {
    // Seed the random number generator
    srand(time(0));

    // Step 1: Read the data
    vector<vector<string>> data;

    cout<<argc<<" arguments provided."<<endl;
    cout<<"argv[0]: "<<argv[0]<<endl;
    cout<<"argv[1]: "<<argv[1]<<endl;
    //get the type of dataset argv[1]
    // cout<<getytpe(argv[1]);
    // cout<<"type "<<argv[1].c_str()<<" to read the dataset."<<endl;
    cout<<(string(argv[1]) == "1")<<endl;

    // Uncomment the line below to read the adult dataset
    if(string(argv[1]) == "2") {
        readCSV("Datasets/adult.data", data);  // Replace this path with the file path on your system
        cout << "Reading Adult dataset." << endl;
        vector<string> attributes = {"age", "workclass", "fnlwgt", "education", "education_num", 
            "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", 
            "capital_loss", "hours_per_week", "native_country", "income"};
        writeAttributes("attribute_list.txt", attributes);
    } else if(string(argv[1]) == "1") {
        cout << "Reading Iris dataset as default." << endl;   
        readCSV("Datasets/Iris.csv", data); // Read the test data as well
        cout<<"size of data: "<<data.size()<<endl;
        vector<string> attributes = data[0];
        data.erase(data.begin());  // Remove the header row
        writeAttributes("attribute_list.txt", attributes);
    }
    else{
        cerr << "Invalid argument. Please use 1 for Adult dataset or 2 for Iris dataset." << endl;
        return 1;
    }

    cout<<data.size()<<endl;
    // Step 2: Shuffle the data randomly
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    // random_shuffle(data.begin(), data.end());

    // Step 3: Split the data into 80% training and 20% testing
    size_t train_size = data.size() * 0.8;
    vector<vector<string>> train_data(data.begin(), data.begin() + train_size);
    vector<vector<string>> test_data(data.begin() + train_size, data.end());

    // Step 4: Write the training and testing datasets to separate CSV files
    writeCSV("training_dataset80.csv", train_data);
    writeCSV("testing_dataset20.csv", test_data);

    cout << "Training and testing datasets have been saved successfully!" << endl;

    return 0;
}
