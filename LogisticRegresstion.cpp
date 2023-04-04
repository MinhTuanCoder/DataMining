#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <typeinfo>
using namespace std;

class LogisticRegression
{
public:
    LogisticRegression(double lr = 0.001, int n_iters = 1000) : lr(lr), n_iters(n_iters) {}
    // dấu 2 chấm để thể hiện gán các giá trị khởi tạo

    void fit(vector<vector<double>> X, vector<int> y)
    {
        // X là ma trận dữ liệu huấn luyện với mỗi hàng là một mẫu và mỗi cột là một đặc trưng
        // y là vector chứa nhãn cho mỗi mẫu trong tập dữ liệu huấn luyện

        int n_samples = X.size();     // số lượng mẫu trong tập dữ liệu huấn luyện
        int n_features = X[0].size(); // số lượng đặc trưng trong tập dữ liệu huấn luyện
        weights.resize(n_features);   // thay đổi kích thước của vector weights để bằng với số lượng đặc trưng
        bias = 0;                     // khởi tạo độ chệch bằng 0

        for (int i = 0; i < n_iters; i++)
        {                                          // lặp lại quá trình cập nhật trọng số và độ chệch n_iters lần
            vector<double> linear_pred(n_samples); // khởi tạo vector chứa tổng tuyến tính của các đặc trưng đầu vào và trọng số
            for (int j = 0; j < n_samples; j++)
            {
                linear_pred[j] = dot(X[j], weights) + bias; // tính tổng tuyến tính cho mỗi mẫu trong tập dữ liệu huấn luyện
            }
            vector<double> predictions = sigmoid(linear_pred); // áp dụng hàm sigmoid cho tổng tuyến tính để tính toán dự đoán

            vector<double> dw(n_features); // khởi tạo vector chứa gradient của hàm mất mát theo trọng số
            for (int j = 0; j < n_features; j++)
            {
                double sum = 0;
                for (int k = 0; k < n_samples; k++)
                {
                    sum += X[k][j] * (predictions[k] - y[k]); // tính toán gradient cho mỗi trọng số
                }
                dw[j] = (1.0 / n_samples) * sum;
            }

            for (int j = 0; j < n_features; j++)
            {
                weights[j] -= lr * dw[j]; // cập nhật trọng số bằng cách sử dụng thuật toán gradient descent
            }
            double db = 0; 
            for (int j = 0; j < n_samples; j++)
            {
                db += predictions[j] - y[j]; // tính toán gradient của hàm mất mát theo độ chệch
            }
            db *= (1.0 / n_samples);
            bias -= lr * db; // cập nhật độ chệch bằng cách sử dụng thuật toán gradient descent
        }
    }

    vector<int> predict(vector<vector<double>> X)
    {
        int n_samples = X.size();              // Số lượng mẫu
        vector<double> linear_pred(n_samples); // Dự đoán tuyến tính
        for (int i = 0; i < n_samples; i++)
        {
            linear_pred[i] = dot(X[i], weights) + bias; // Tính tích vô hướng của mỗi hàng trong X với vector trọng số và cộng thêm hệ số chệch
        }
        vector<double> y_pred = sigmoid(linear_pred); // Đưa dự đoán tuyến tính qua hàm sigmoid để có được xác suất dự đoán
        vector<int> class_pred(n_samples);            // Dự đoán lớp
        for (int i = 0; i < n_samples; i++)
        {
            class_pred[i] = y_pred[i] <= 0.5 ? 0 : 1; // Chuyển đổi xác suất thành dự đoán lớp bằng cách gán 0 nếu xác suất nhỏ hơn hoặc bằng 0.5 và 1 nếu ngược lại
        }
        return class_pred;
    }

    vector<double> get_weights()
    {
        return this->weights;
    }
    double get_bias()
    {
        return this->bias;
    }

private:
    double lr;              // Tốc độ học
    int n_iters;            // Số lần lặp
    vector<double> weights; // Vector trọng số
    double bias;            // Hệ số chệch

    double dot(vector<double> a, vector<double> b)
    { // Hàm tính tích vô hướng của hai vector
        double result = 0;
        for (int i = 0; i < a.size(); i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    vector<double> sigmoid(vector<double> x)
    { // Hàm tính sigmoid của các phần tử trong vector
        vector<double> result(x.size());
        for (int i = 0; i < x.size(); i++)
        {
            result[i] = 1 / (1 + exp(-x[i]));
        }
        return result; // Trả về vector chứa giá trị sigmoid của từng phần tử trong x
    }
};


vector<vector<string>> read_csv(string nameFile, int numRow)
{
    vector<vector<string>> data;
    ifstream file(nameFile);
    string line;
    int count = 0;
    while (getline(file, line))
    {
        if (count < numRow)
        {
            vector<string> row;
            stringstream lineStream(line);
            string cell;
            while (getline(lineStream, cell, ','))
            {
                // Lo?i b? d?u ngo?c k�p kh?i �
                std::string result;
                for (char c : cell)
                {
                    if (c != '\"')
                    {
                        result += c;
                    }
                }
                cell = result;
                row.push_back(cell);
            }
            data.push_back(row);
            count++;
        }
    }
    return data;
}
void one_hot_encoding(vector<vector<string>> &input,int nrow)
{
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < 17; j++)
        {
            if (input[i][j] == "management" || input[i][j] == "married" || input[i][j] == "tertiary" || input[i][j] == "yes" || input[i][j] == "cellular" || input[i][j] == "feb" || input[i][j] == "success")
                input[i][j] = "1";
            else if (input[i][j] == "technician" || input[i][j] == "single" || input[i][j] == "secondary" || input[i][j] == "telephone" || input[i][j] == "mar" || input[i][j] == "failure")
                input[i][j] = "2";
            else if (input[i][j] == "unknown" || input[i][j] == "no" || input[i][j] == "jan")
                input[i][j] = "0";
            else if (input[i][j] == "retired" || input[i][j] == "divorced" || input[i][j] == "primary" || input[i][j] == "apr" || input[i][j] == "other")
                input[i][j] = "3";
            else if (input[i][j] == "entrepreneur" || input[i][j] == "may")
                input[i][j] = "4";
            else if (input[i][j] == "blue-collar" || input[i][j] == "jun")
                input[i][j] = "5";
            else if (input[i][j] == "admin." || input[i][j] == "jul")
                input[i][j] = "6";
            else if (input[i][j] == "services" || input[i][j] == "aug")
                input[i][j] = "7";
            else if (input[i][j] == "self-employed" || input[i][j] == "sep")
                input[i][j] = "8";
            else if (input[i][j] == "unemployed" || input[i][j] == "oct")
                input[i][j] = "9";
            else if (input[i][j] == "housemaid" || input[i][j] == "nov")
                input[i][j] = "10";
            else if (input[i][j] == "student" || input[i][j] == "dec")
                input[i][j] = "11";
        }
    }
}
//Ham tinh accuracy
double accuracy(vector<int> y_test, vector<int> y_predict){
	double total=y_test.size();
	double count = 0;
	for(int i=0;i<total;i++){
		if(y_test[i] == y_predict[i])
			count++;
	}
	return count/total;
}

int main()
{
    string namefile = "train.csv";
    int nrow = 45211;
    vector<vector<string>> df = read_csv(namefile, nrow);
    one_hot_encoding(df,nrow); // chuan hoa
    vector<vector<double>> data_train(df.size()); // chuyen string sang double
    for (int i = 0; i < df.size(); i++)
    {
        data_train[i].resize(df[i].size());
        for (int j = 0; j < df[i].size(); j++)
        {
            data_train[i][j] = std::stod(df[i][j]);
        }
    }

    int n_samples = data_train.size();
    int n_features = data_train[0].size() - 1;
    vector<vector<double>> x_train(n_samples, vector<double>(n_features));
    vector<int> y_train(n_samples);

    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            x_train[i][j] = data_train[i][j];
        }
        y_train[i] = data_train[i][n_features];
    }
    ////////////////////////////
    string namefile_test = "test.csv";
    int nrow_test = 4521;
    vector<vector<string>> df_test = read_csv(namefile_test, nrow_test);
    one_hot_encoding(df_test,nrow_test); // Chu?n h�a

    vector<vector<double>> data_test(df_test.size()); // chuyen string sang double
    for (int i = 0; i < df_test.size(); i++)
    {
        data_test[i].resize(df_test[i].size());
        for (int j = 0; j < df_test[i].size(); j++)
        {
            data_test[i][j] = std::stod(df_test[i][j]);
        }
    }

    int n_samples_test = data_test.size();
    int n_features_test = data_test[0].size() - 1;
    vector<vector<double>> x_test(n_samples_test, vector<double>(n_features_test));
    vector<int> y_test(n_samples_test);

    for (int i = 0; i < n_samples_test; i++)
    {
        for (int j = 0; j < n_features_test; j++)
        {
            x_test[i][j] = data_test[i][j];
        }
        y_test[i] = data_test[i][n_features_test];
    }
//        for(int i=0;i<y_test.size();i++){ // in ra  y_test
//    	cout<<y_test[i]<<" ";
//	}
    //////////////////////////////////////////////
    LogisticRegression model = LogisticRegression(0.01, 1000);
    model.fit(x_train, y_train);
    cout << "Vector weights: ";
    for (int i = 0; i < model.get_weights().size(); i++)
    {
        cout << model.get_weights()[i] << " ";
    }
    cout << "\nBias: " << model.get_bias();

    vector<int> y_predict = model.predict(x_test);

	cout<<"\nAccuracy: "<<accuracy(y_test,y_predict);
    return 0;
}
