#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

double sum(vector<double> vec) {
    double sum = 0;
    for (double d : vec) {
        sum += d;
    }
    return sum;
}

double mean(vector<double> vec) {
    return sum(vec) / vec.size();
}

double median(vector<double> vec) {
    int size = vec.size();
    double med;

    if (size % 2 == 0)
        med = (vec[size / 2 - 1] + vec[size / 2]) / 2.0;
    else
        med = vec[size / 2];

    return med;
}

double range(vector<double> vec) {
    if (vec.empty())
        return 0;
    
    double min = vec[0];
    double max = vec[0];

    for (int i = 0; i < vec.size(); i++) {
        double elem = vec[i];
        if (elem < min)
            min = elem;
        if (elem > max)
            max = elem;
    }

    return max - min;
}

double covar(vector<double> v1, vector<double> v2) {
    double v1_mean = mean(v1);
    double v2_mean = mean(v2);
    double sum_prod_devs = 0;
    int size = fmin(v1.size(), v2.size());

    for (int i = 0; i < size; i++) {
        sum_prod_devs = (v1[i] - v1_mean) * (v2[i] - v2_mean);
    }

    return sum_prod_devs / (size - 1);
}

double cor(vector<double> v1, vector<double> v2) {
    double v1_std = sqrt(covar(v1, v1));
    double v2_std = sqrt(covar(v2, v2));
    return covar(v1, v2) / (v1_std * v2_std);
}

void print_stats(vector<double> vec) {
    cout << "sum:    " << sum(vec) << endl;
    cout << "mean:   " << mean(vec) << endl;
    cout << "median: " << median(vec) << endl;
    cout << "range:  " << range(vec) << endl;
}

int main(int argc, char **argv) {
    ifstream inFS;
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    cout << "Opening file Boston.csv" << endl;

    inFS.open("Boston.csv");
    if (!inFS.is_open()) {
        cout << "Could not open file Boston.csv" << endl;
        return 1;
    }

    cout << "Reading line 1" << endl;
    getline(inFS, line);

    cout << "heading: " << line << endl;

    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length " << rm.size() << endl;
    inFS.close();

    cout << "\nStats for rm" << endl;
    print_stats(rm);

    cout << "\nStats for medv" << endl;
    print_stats(medv);

    cout << "\n Covariance = " << covar(rm, medv) << endl;
    cout << "\n Correlation = " << cor(rm, medv) << endl;
    cout << "\nProgram terminated";
    
    return 0;
}