#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace chrono;

const double learning_rate = 0.001;
const double epsilon = 0.000001;

// returns the specified column of the matrix
vector<double> get_column(const vector<vector<double>> &df, int col) {
    vector<double> column(df.size());

    for (int i = 0; i < df.size(); i++) {
        column[i] = df[i][col];
    }

    return column;
}

// returns vector of prior probabilities of each category in df[ycol]
// assumes that categories are numbered 0 through n
vector<double> prior(const vector<vector<double>> &df, int ycol) {
    vector<int> category_counts;

    // count examples
    for (int i = 0; i < df.size(); i++) {
        int val = (int) df[i][ycol];
        
        if (category_counts.size() < val + 1)
            category_counts.resize(val + 1, 0);

        category_counts[val]++;
    }

    vector<double> res;

    // divide examples by total
    for (int i = 0; i < category_counts.size(); i++) {
        res.push_back(category_counts[i] / (double) df.size());
    }

    return res;
}

// returns conditional probabilities for the given xcols in df
//  vector is structured as:
//      property -> value in df[ycol] -> conditional probability
// params:
//  df: data
//  xcols: list of indices of columns of discrete, categorical predictors
//      assume category values are numbered from 0-n
//  ycol: target column index
//  num_target_categories: number of unique values in target column
vector<vector<vector<double>>> disc_lh(
    const vector<vector<double>> &df,
    const vector<int> &xcols,
    int ycol,
    int num_target_categories
) {
    vector<vector<vector<double>>> res(
        xcols.size(),
        vector<vector<double>>(num_target_categories)
    );

    // counting occurrences of values
    for (int i = 0; i < df.size(); i++) {
        // looping through rows
        int target_col_val = df[i][ycol];

        for (int j = 0; j < xcols.size(); j++) {
            // looping through discrete attr columns
            double val = df[i][xcols[j]];

            // resize vectors if observed val is too large
            if (res[j][0].size() < val + 1){
                for (int k = 0; k < num_target_categories; k++) {
                    res[j][k].resize(val + 1);
                }
            }

            res[j][target_col_val][val]++;
        }
    }

    // divide occurences by totals
    for (int i = 0; i < res.size(); i++) {
        // looping through table of cond probabilities for each attr
        for (int j = 0; j < num_target_categories; j++) {
            // looping through possible values of df[ycol]
            int total = 0;

            // counting conditional totals
            for (int k = 0; k < res[i][j].size(); k++) {
                total += res[i][j][k];
            }

            // dividing occurrences by totals
            for (int k = 0; k < res[i][j].size(); k++) {
                res[i][j][k] = res[i][j][k] / (double) total;
            }
        }
    }

    return res;
}

// returns mean value of a column xcol for different values if column ycol
vector<double> cond_means(
    const vector<vector<double>> &df,
    int xcol, 
    int ycol,
    int num_target_categories
) {
    vector<double> res(num_target_categories);
    vector<double> counts(num_target_categories);

    // summing values
    for (int i = 0; i < df.size(); i++) {
        int target_col_val = df[i][ycol];
        
        res[target_col_val] += df[i][xcol];
        counts[target_col_val]++;
    }

    // dividing sums by totals
    for (int i = 0; i < num_target_categories; i++) {
        res[i] = res[i] / (double) counts[i];
    }

    return res;
}

// returns variance of a column given value in ycol
vector<double> cond_variances(
    const vector<vector<double>> &df,
    int xcol,
    int ycol,
    int num_target_categories,
    const vector<double> &cond_means
) {
    vector<double> res(num_target_categories);
    vector<double> counts(num_target_categories);

    // summing variances
    for (int i = 0; i < df.size(); i++) {
        int target_col_val = df[i][ycol];
        
        res[target_col_val] += pow(df[i][xcol] - cond_means[target_col_val], 2);
        counts[target_col_val]++;
    }

    // averaging variances
    for (int i = 0; i < num_target_categories; i++) {
        res[i] = res[i] / (double) counts[i];
    }

    return res;
}

// returns likelihood of a continuous attribute having a given value 
double cont_lh(double val, double mean, double variance) {
    return 1 / sqrt(2 * M_PI * variance) * exp(-pow((val-mean), 2)/(2 * variance));
}

// returns a tuple:
//  vector<double>: priors
//  vector<vector<vector<double>>>: likelihoods for discrete attributes
//  vector<vector<vector<double>>>: conditional means/variances for quantitative attributes
// params:
//  df: data matrix
//  disc_cols: list of indices for discrete attr columns
//  cont_cols: list of indices for continuous attr columns
//  ycol: index of target column
tuple<vector<double>, vector<vector<vector<double>>>, vector<vector<vector<double>>>>
naive_bayes(
    const vector<vector<double>> &df,
    const vector<int> &disc_cols,
    const vector<int> &cont_cols,
    int ycol
) {
    // getting priors
    vector<double> priors = prior(df, ycol);
    // getting likelihoods for discrete data
    vector<vector<vector<double>>> disc_lhs = disc_lh(df,  disc_cols, ycol, priors.size());
    vector<vector<double>> means;
    vector<vector<double>> variances;

    // getting likelihoods for continuous data
    for (int i = 0; i < cont_cols.size(); i++) {
        means.push_back(cond_means(df, cont_cols[i], ycol, priors.size()));
        variances.push_back(cond_variances(df, cont_cols[i], ycol, priors.size(), means[i]));
    }

    // returning model as tuple of vectors
    return {
        priors,
        disc_lhs,
        {means, variances}
    };
}

// returns vector of probability of getting each possible value of target col
// params:
//  priors: list of priors obtained from prior()
//  disc_xvals: list of values for dscrete attrs
//  cond_probs: vector of conditional probabilities obtained from disc_lh()
//  cont_xvals: list of values for continuous attrs
//  cond_mean_vars: list of conditional properties
//      obtained by combining results of cond_means and cond_variances into
//      one 3D matrix
vector<double> calc_raw_probs(
    const vector<double> &priors,
    const vector<double> &disc_xvals,
    const vector<vector<vector<double>>> &cond_probs,
    const vector<double> &cont_xvals,
    const vector<vector<vector<double>>> &cond_means_vars
) {
    vector<double> res;

    // extracting lists of means and lists of variances
    vector<vector<double>> cond_means = cond_means_vars[0];
    vector<vector<double>> cond_variances = cond_means_vars[1];

    double denominator = 0;

    // calculating numerators
    for (int i = 0; i < priors.size(); i++) {
        // looping through possible vals of target column
        double num = 1;

        // product of likelihoods for discrete attrs
        for (int j = 0; j < cond_probs.size(); j++) {
            num *= cond_probs[j][i][disc_xvals[j]];
        }

        // product of likelihoods for continuous attrs
        for (int j = 0; j < cond_means.size(); j++) {
            num *= cont_lh(cont_xvals[j], cond_means[j][i], cond_variances[j][i]);
        }

        num *= priors[i];
        denominator += num;
        res.push_back(num);
    }

    // dividing numerators by denominators to get probability
    for (int i = 0; i < res.size(); i++) {
        res[i] = res[i] / denominator;
    }

    return res;
}

// returns vector of predictions given a dataset and model
//  for each row, contains probabilities for each possible value of target col
// params:
//  nb: model obtained from naive_bayes()
//  df: data matrix
//  disc_cols: list of indices for discrete attr columns
//  cont_cols: list of indices for continuous attr columns
//  ycol: index of target column
vector<vector<double>> predict(
    const tuple<vector<double>, vector<vector<vector<double>>>, vector<vector<vector<double>>>> &nb,
    const vector<vector<double>> &df,
    const vector<int> &disc_cols,
    const vector<int> &cont_cols,
    int ycol
) {
    vector<vector<double>> res;

    // extracting elements of model
    vector<double> priors = get<0>(nb);
    vector<vector<vector<double>>> disc_lhs = get<1>(nb);
    vector<vector<vector<double>>> cont_lhs = get<2>(nb);

    for (int i = 0; i < df.size(); i++){
        // looping through rows of data
        vector<double> disc_vals;
        vector<double> cont_vals;

        // getting vals in discrete attr columns
        for (int j = 0; j < disc_cols.size(); j++) {
            disc_vals.push_back(df[i][disc_cols[j]]);
        }

        // getting vals in continuous attr columns
        for (int j = 0; j < cont_cols.size(); j++) {
            cont_vals.push_back(df[i][cont_cols[j]]);
        }

        // getting probabilities of observing target values
        res.push_back(calc_raw_probs(priors, disc_vals, disc_lhs, cont_vals, cont_lhs));
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
        else
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

// prints model model obtained from naive_bayes()
// headers must be in same order as the original columns in data matrix
void print_nb_model(
    const tuple<vector<double>, vector<vector<vector<double>>>, vector<vector<vector<double>>>> &nb,
    const vector<string> &headers,
    const vector<int> &disc_xcols,
    const vector<int> &cont_xcols
) {
    // extracting elements of model
    vector<double> priors = get<0>(nb);
    vector<vector<vector<double>>> disc_lhs = get<1>(nb);
    vector<vector<vector<double>>> cont_lhs = get<2>(nb);

    cout << "A-priori Probabilities:" << endl;

    for (int i = 0; i < priors.size(); i++) {
        cout << right << setw(8) << i << " ";
    }

    cout << endl;

    // printing priors
    for (int i = 0; i < priors.size(); i++) {
        cout << right << setw(8) << priors[i] << " ";
    }

    cout << endl << endl;

    cout << "Conditional Probabiliites:" << endl;

    // printing conditional probability tables for discrete attrs
    for (int i = 0; i < disc_lhs.size(); i++) {
        cout << "\t" << headers[disc_xcols[i]] << endl;
        cout << "\t  ";

        for (int j = 0; j < disc_lhs[i][0].size(); j++) {
            cout << right << setw(8) << j << " ";
        }

        cout << endl;

        for (int j = 0; j < priors.size(); j++) {
            cout << "\t" << j << " ";

            for (int k = 0; k < disc_lhs[i][j].size(); k++) {
                cout << right << setw(8) << disc_lhs[i][j][k] << " ";
            }

            cout << endl;
        }

        cout << endl;
    }

    cout << "Means and Variances" << endl;

    // printing means and variances for continuous attrs
    for (int i = 0; i < cont_lhs[0].size(); i++) {
        cout << "\t" << headers[cont_xcols[i]] << endl;
        cout << "\t  " << right << setw(8) << "Mean" << " " << right << setw(8) << "Variance" << endl;
        
        for (int j = 0; j < priors.size(); j++) {
            cout << "\t" << j << " ";
            cout << right << setw(8) << cont_lhs[0][i][j] << " " << right << setw(8) << cont_lhs[1][i][j] << endl;
        }
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
        for (int i = 0; line_stream.good(); i++) {
            getline(line_stream, col, ',');
            int val = stoi(col);

            // make p-class values 0-based
            if (i == 0) {
                val--;
            }

            row.push_back(val);
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
    vector<int> disc_xcols = {0, 2};
    vector<int> cont_xcols = {3};
    time_point<system_clock> start = system_clock::now();
    auto model = naive_bayes(train, disc_xcols, cont_xcols, 1);
    time_point<system_clock> end = system_clock::now();
    duration<double> elapsed_time = end - start;

    cout << "Training took " << elapsed_time.count() << "s" << endl << endl;

    // print model
    print_nb_model(model, headers, disc_xcols, cont_xcols);
    cout << endl;

    // test model
    vector<vector<double>> predictions = predict(model, test, disc_xcols, cont_xcols, 1);
    vector<double> pred_survived = get_column(predictions, 1);

    // print metrics
    print_metrics(get_column(test, 1), pred_survived);
    cout << endl;

    return 0;
}
