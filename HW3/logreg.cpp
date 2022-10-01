#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace chrono;

const double learning_rate = 0.001;
const double epsilon = 0.00001;

// returns the specified column of the matrix
vector<double> get_column(const vector<vector<double>> &df, int col) {
    vector<double> column(df.size());

    for (int i = 0; i < df.size(); i++) {
        column[i] = df[i][col];
    }

    return column;
}

// returns transpose of a 2D matrix
vector<vector<double>> transpose(const vector<vector<double>> &m)
{
    vector<vector<double>> t(m[0].size(), vector<double>(m.size()));

    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[0].size(); j++)
        {
            t[j][i] = m[i][j];
        }
    }

    return t;
}

// returns product of 2D matrix with a vector
vector<double> matrix_mult(const vector<vector<double>> &a, const vector<double> &b)
{
    vector<double> res(a.size());

    for (int i = 0; i < a.size(); i++)
    {
        double sum = 0;

        for (int j = 0; j < b.size(); j++)
        {
            sum += a[i][j] * b[j];
        }

        res[i] = sum;
    }

    return res;
}

// return dot product of two vectors
double matrix_mult(const vector<double> &a, const vector<double> &b)
{
    double sum = 0;

    for (int i = 0; i < a.size(); i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

// returns product of 2D matrix with a scalar
vector<vector<double>> matrix_mult(const vector<vector<double>> &a, double b)
{
    vector<vector<double>> res(a.size(), vector<double>(a[0].size()));

    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            res[i][j] = a[i][j] * b;
        }
    }

    return res;
}

// returns product of a vector with a scalar
vector<double> matrix_mult(const vector<double> &a, double b)
{
    vector<double> res(a.size());

    for (int i = 0; i < a.size(); i++)
    {
        res[i] = a[i] * b;
    }

    return res;
}

// returns result of subtracting vector b from vector a
vector<double> vector_sub(const vector<double> &a, const vector<double> &b)
{
    vector<double> res(a.size());

    for (int i = 0; i < a.size(); i++)
    {
        res[i] = a[i] - b[i];
    }

    return res;
}

// calculates vector of log odds with given weights
vector<double> sigmoid(const vector<vector<double>> &x, const vector<double> &w)
{
    vector<double> y(x.size());

    for (int i = 0; i < x.size(); i++)
    {
        y[i] = 1 / (1 + pow(M_E, -matrix_mult(x[i], w)));
    }

    return y;
}

// training for logistic regression
// continues training until change in weights is less than epsilon
// returns vector of optimal weights
vector<double> gradient_descent(const vector<vector<double>> &x, const vector<double> &y)
{
    vector<double> w(x[0].size());
    vector<vector<double>> x_transpose = transpose(x);

    while (true)
    {
        vector<double> w_prev = w;

        // calculate new weights from prev iteration
        w = vector_sub(
            w,
            matrix_mult(
                matrix_mult(
                    x_transpose,
                    vector_sub(
                        sigmoid(x, w),
                        y
                    )
                ),
                learning_rate
            )
        );

        bool should_break = true;

        for (int i = 0; i < w.size(); i++)
        {
            if (abs(w[i] - w_prev[i]) > epsilon) {
                // change in weights was too big
                // must continue training
                should_break = false;
                break;
            }
        }

        if (should_break)
            break;
    }

    return w;
}

// returns weights for a logistic regression model
// params:
//  df: complete matrix of dataset
//  xcols: list of indices of columns in df to base predictions on
//  ycol: index of target column in df
vector<double> logistic_regression(const vector<vector<double>> &df, const vector<int> &xcols, int ycol)
{
    vector<vector<double>> x(df.size(), vector<double>(xcols.size() + 1, 1));
    vector<double> y(df.size());

    for (int i = 0; i < df.size(); i++)
    {
        // extract x matrix
        for (int j = 0; j < xcols.size(); j++)
        {
            x[i][j + 1] = df[i][xcols[j]];
        }

        // extract y vector
        y[i] = df[i][ycol];
    }

    // perform gradient descent to get w vector
    return gradient_descent(x, y);
}

// returns vector of predictions given a dataset and model
vector<double> predict(const vector<vector<double>> &df, const vector<int> &xcols, const vector<double> &w) {
    vector<vector<double>> x(df.size(), vector<double>(xcols.size() + 1, 1));

    // extract x matrix
    for (int i = 0; i < df.size(); i++)
    {
        for (int j = 0; j < xcols.size(); j++)
        {
            x[i][j + 1] = df[i][xcols[j]];
        }
    }

    vector<double> log_likelihoods = matrix_mult(x, w);
    vector<double> res(log_likelihoods.size());

    // convert log odds to probabilities
    for (int i = 0; i < log_likelihoods.size(); i++) {
        res[i] = exp(log_likelihoods[i]) / (1 + exp(log_likelihoods[i]));
    }

    return res;
}

// given truth and predictions,
// returns confusion matrix in vector form
// order is TP, FP, TN, FN
vector<int> confusion_matrix(const vector<double> &y, const vector<double> &y_pred) {
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;

    for (int i = 0; i < y.size(); i++) {
        if (y[i] == 1 && y_pred[i] >= 0.5)
            tp++;
        else if (y[i] == 0 && y_pred[i] >= 0.5)
            fp++;
        else if (y[i] == 0 && y_pred[i] < 0.5)
            tn++;
        else if (y[i] == 1 && y_pred[i] < 0.5)
            fn++;
    }

    return {tp, fp, tn, fn};
}

// calculates accuracy of a model
double accuracy(const vector<double> &y, const vector<double> &y_pred) {
    vector<int> conf_matrix = confusion_matrix(y, y_pred);
    return (conf_matrix[0] + conf_matrix[2]) / (double) y.size();
}

// calculates sensitivity of a model
double sensitivity(const vector<double> &y, const vector<double> &y_pred) {
    vector<int> conf_matrix = confusion_matrix(y, y_pred);
    return (conf_matrix[0]) / (double) (conf_matrix[0] + conf_matrix[3]);
}

// calculates sensitivity of a model
double specificity(const vector<double> &y, const vector<double> &y_pred) {
    vector<int> conf_matrix = confusion_matrix(y, y_pred);
    return (conf_matrix[2]) / (double) (conf_matrix[2] + conf_matrix[1]);
}

// prints model coefficients in tabular format
void print_coefficients(const vector<double> &w, const vector<string> &headers, const vector<int> &xcols) {
    cout << right << setw(25) << "Coefficient" << endl;
    cout << left << setw(15) << "(intercept)"
        << right << setw(10) << w[0] << endl;
    for (int i = 1; i < w.size(); i++) {
        cout << left << setw(15) << headers[xcols[i - 1]]
            << right << setw(10) << w[i] << endl;
    }
}

// prints model metrics in tabular format
void print_metrics(const vector<double> &y, const vector<double> &y_pred) {
    vector<string> row_names = {"accuracy", "sensitivity", "specificity"};
    vector<double> row_vals = {accuracy(y, y_pred), sensitivity(y, y_pred), specificity(y, y_pred)};
    for (int i = 0; i < row_names.size(); i++) {
        cout << left << setw(15) << row_names[i]
            << right << setw(10) << row_vals[i] << endl;
    }
}

int main(int argc, char **argv)
{
    ifstream inFS;
    string line;
    vector<vector<double>> train;
    vector<vector<double>> test;
    const string data_file = "titanic_project.csv";

    cout << "Opening file " << data_file << endl;

    inFS.open(data_file);
    if (!inFS.is_open())
    {
        cout << "Could not open file " << data_file << endl;
        return 1;
    }

    // process and store column headers
    cout << "Reading line 1" << endl;
    getline(inFS, line);

    stringstream line_stream(line);
    vector<string> headers;
    string col;

    // skip id header
    getline(line_stream, col, ',');

    while (line_stream.good()) {
        getline(line_stream, col, ',');
        headers.push_back(col);
    }

    cout << "heading: " << line << endl << endl;

    int num_observations = 0;
    while (inFS.good())
    {
        getline(inFS, line);
        line_stream = stringstream(line);
        vector<double> row;

        // skip index col
        getline(line_stream, col, ',');

        // read data
        while (line_stream.good()) {
            getline(line_stream, col, ',');
            row.push_back(stoi(col));
        }

        if (num_observations < 800)
        {
            train.push_back(row);
        }
        else
        {
            test.push_back(row);
        }

        num_observations++;
    }

    // make model and measure training time
    vector<int> xcols = {2};
    time_point<system_clock> start = system_clock::now();
    vector<double> w = logistic_regression(train, xcols, 1);
    time_point<system_clock> end = system_clock::now();
    duration<double> elapsed_time = end - start;

    cout << "Training took " << elapsed_time.count() << "s" << endl << endl;

    // print model
    print_coefficients(w, headers, xcols);
    cout << endl;

    // test model
    vector<double> predictions = predict(test, xcols, w);

    // print metrics
    print_metrics(get_column(test, 1), predictions);
    cout << endl;

    return 0;
}
