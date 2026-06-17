#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <random>
#include <filesystem>
#include <algorithm>
#include <functional>
#include <chrono>

using namespace std;

#ifdef USE_FLOAT
    #define TYPE float
#else
    #define TYPE double
#endif


template <typename T>
size_t argMax(const vector<T>& vec){
    size_t idx = 0;
    TYPE max_val = vec[0]; 
    for (size_t i = 1; i < vec.size(); i++){
        if (max_val < vec[i]){
            max_val = vec[i];
            idx = i;
        }
    }
    return idx;
}

// 文字分割
vector<string> split(const string text, const char delimiter){
    vector<string> columns;
    if (text.empty()){
        return columns;
    }
    stringstream stream{text};
    string buff;
    while (getline(stream, buff, delimiter)){
        columns.push_back(buff);
    }
    return columns;
}

vector<TYPE> generateRandomVector(int n, int random_state){
    mt19937 mt(random_state);
    uniform_real_distribution<TYPE> dist((TYPE)0.0, (TYPE)1.0);
    vector<TYPE> out(n);
    for (int i = 0; i < n; i++){
        out[i] = dist(mt);
    }
    return out;
}

void createDataset(string filePath, 
                   vector<pair<vector<int>, int>>& dataset, 
                   map<string, int>& wordToId, 
                   vector<vector<TYPE>>& wordVectors,
                   map<string, int>& labelToId,
                   vector<vector<TYPE>>& labelVectors,
                   vector<string>& idToLabel,
                   int vectorSize = 1000) {

    if (!filesystem::exists(filePath)){
        cerr << "ファイルが存在しません: " << filePath << endl;
        return;
    }

    cout << "INPUT FILE " << filePath << endl;
    ifstream file(filePath);
    string line;
    hash<string> hasher;

    int wordIdCounter = 0;
    int labelIdCounter = 0;

    while (getline(file, line)){
        vector<string> rows = split(line, ' ');
        if (rows.empty()) continue;

        // ラベルの処理
        string labelStr = rows[0];
        if (labelToId.find(labelStr) == labelToId.end()){
            labelToId[labelStr] = labelIdCounter++;
            idToLabel.push_back(labelStr); // IDからラベル名に戻す用
        }
        int labelId = labelToId[labelStr];

        vector<int> articleWordIds;
        for (int i = 1; i < rows.size(); i++){
            string wordStr = rows[i];
            
            if (wordToId.find(wordStr) == wordToId.end()){
                wordToId[wordStr] = wordIdCounter++;
                
                vector<TYPE> vec(vectorSize, (TYPE)0.0);
                size_t hashValue = hasher(wordStr);
                mt19937 mt(hashValue);
                uniform_real_distribution<TYPE> dist((TYPE)-1.0, (TYPE)1.0);

                for (int k = 0; k < vectorSize; k++){
                    vec[k] = dist(mt);
                }
                wordVectors.push_back(vec);
            }
            
            articleWordIds.push_back(wordToId[wordStr]);
        }
        dataset.push_back({articleWordIds, labelId});
    }

    // ラベルのOne-hotベクトルの作成
    labelVectors.assign(labelIdCounter, vector<TYPE>(labelIdCounter, (TYPE)0.0));
    for (int i = 0; i < labelIdCounter; i++){
        labelVectors[i][i] = (TYPE)1.0;
    }
}

int step(TYPE x, TYPE boarder = (TYPE)0.0){
    if (x >= boarder){
        return 1;
    } else {
        return 0;
    }
}

struct Matrix{
    int col, row;
    vector<TYPE> data;
    Matrix(int col, int row, const vector<TYPE>& vec) : col(col), row(row){
        data.assign(col * row, (TYPE)0.0);
        if (data.size() != vec.size()){
            cerr << "入力サイズがあっていません" << endl;
            return;
        }
        data = vec;
    }
    Matrix(int col, int row) : col(col), row(row){
        data.assign(col * row, (TYPE)0.0);
    }
    Matrix(){};
};

Matrix mulMat(const Matrix& a, const Matrix& b){
    if (a.col != b.row){
        cerr << "入力サイズがあっていません" << endl;
    }
    int col = b.col;
    int row = a.row;
    Matrix out(col, row);

    #pragma omp parallel for
    for (int i = 0; i < row; i++){
        for (int k = 0; k < a.col; k++){
            for (int j = 0; j < col; j++){
                out.data[col * i + j] += a.data[a.col * i + k] * b.data[b.col * k + j];
            }
        }
    }
    return out;
};

struct Perceptron{
    int nIter;
    TYPE eta; 
    int randomState;
    int classSize;

    Matrix w_;

    // 
    Perceptron(int nIter=50, TYPE eta=(TYPE)0.1, int classSize=2, int randomState=1) 
        : nIter(nIter), eta(eta), randomState(randomState), classSize(classSize){};

    void init_weight(int col, int row, int n){
        w_.col = col;
        w_.row = row;
        w_.data = generateRandomVector(n, randomState);
    }

    Matrix inputNet(const Matrix& x){
        if (w_.data.size() == 0){
            cout << "initilize weight" << endl;
            init_weight(x.row, classSize, x.row * classSize);
        }
        return mulMat(w_, x);
    }

    Matrix activation(const Matrix& x){
        Matrix out(x.col, x.row);
        for (int i = 0; i < x.row; i++){
            out.data[out.col * i] = step(x.data[x.col * i]);    
        }
        return out;
    }

    Matrix loss(const Matrix& x, const vector<TYPE>& labelOnehot){
        Matrix out(x.col, x.row);
        for (int i = 0; i < x.row; i++){
            out.data[i] = labelOnehot[i] - x.data[i];
        }
        return out;
    }

    Matrix forward(const Matrix& x){
        Matrix x_ = inputNet(x);
        x_ = activation(x_);
        return x_;
    }

    void updateWeight(const Matrix& x, const Matrix& loss){
        for (int i = 0; i < w_.row; i++){
            for (int j = 0; j < w_.col; j++){
                w_.data[w_.col * i + j] += eta * loss.data[i] * x.data[j];
            }
        }
    }

    void fit(const vector<pair<vector<int>, int>>& dataset, const vector<vector<TYPE>>& wordVectors, const vector<vector<TYPE>>& labelVectors){
        for (int i = 0; i < nIter; i++){
            for (const auto& data : dataset){
                const vector<int>& articleWordIds = data.first;
                int labelId = data.second;

                vector<TYPE> docVec(1000, (TYPE)0.0);
                int validWordCount = 0;

                for (int wordId : articleWordIds){
                    const vector<TYPE>& wordVec = wordVectors[wordId];
                    for (int k = 0; k < 1000; k++){
                        docVec[k] += wordVec[k];
                    }
                    validWordCount++;
                }

                if (validWordCount > 0){
                    for (int k = 0; k < 1000; k++){
                        docVec[k] /= (TYPE)validWordCount;
                    }
                }

                Matrix x0(1, docVec.size(), docVec);
                Matrix x_ = forward(x0);
                Matrix loss_ = loss(x_, labelVectors[labelId]);

                updateWeight(x0, loss_);
            }
        }
    }

    // --- 高速化：テスト時も文字検索を最小限にする ---
    void test(string testFilePath, const map<string, int>& trainWordToId, const vector<vector<TYPE>>& wordVectors, const map<string, int>& labelToId, const vector<string>& idToLabel){
        if (!filesystem::exists(testFilePath)){
            cerr << "ファイルが存在しません" << endl;
            return;
        }
        cout << "TEST FILE " << testFilePath << endl;
        
        ifstream file(testFilePath);
        string line;
        
        // テスト結果の集計用
        int numClasses = idToLabel.size();
        vector<int> labelCount(numClasses, 0);
        vector<int> answerCount(numClasses, 0);
        vector<int> correctCount(numClasses, 0);

        while (getline(file, line)){
            vector<string> rows = split(line, ' ');
            if (rows.empty()) continue;
            
            string trueLabelStr = rows[0];
            if (labelToId.find(trueLabelStr) == labelToId.end()) continue; // 学習時にないラベルは無視
            int trueLabelId = labelToId.at(trueLabelStr);

            labelCount[trueLabelId]++;

            vector<TYPE> docVec(1000, (TYPE)0.0); 
            int validWordCount = 0;
            
            for (int i = 1; i < rows.size(); i++){
                string wordStr = rows[i];
                if (trainWordToId.find(wordStr) != trainWordToId.end()){
                    int wordId = trainWordToId.at(wordStr);
                    const vector<TYPE>& wordVec = wordVectors[wordId];
                    for (int k = 0; k < 1000; k++){
                        docVec[k] += wordVec[k];
                    }
                    validWordCount++;
                }
            }

            if (validWordCount > 0){
                for (int k = 0; k < 1000; k++){
                    docVec[k] /= (TYPE)validWordCount;
                }
            } else {
                continue; 
            }

            Matrix inputMatrix(1, docVec.size(), docVec);
            Matrix raw_scores = inputNet(inputMatrix);

            int predictedLabelId = argMax(raw_scores.data);

            if (predictedLabelId == trueLabelId){
                correctCount[trueLabelId]++;
            }
            answerCount[predictedLabelId]++;
        }

        // 結果出力
        string boarder(30, '-');
        cout << boarder << endl;
        for (int i = 0; i < numClasses; i++){
            string labelStr = idToLabel[i];
            
            double precision = (answerCount[i] > 0) ? (double)correctCount[i] / answerCount[i] : 0.0;
            double recall = (labelCount[i] > 0) ? (double)correctCount[i] / labelCount[i] : 0.0;
            
            double fmeasure = 0.0;
            if (precision + recall > 0.0) {
                fmeasure = 2 * precision * recall / (precision + recall); 
            }

            cout << "Label: " << labelStr << endl;
            cout << "  Precision: " << precision << endl;
            cout << "  Recall:    " << recall << endl;
            cout << "  F-measure: " << fmeasure << endl;
        }
        cout << boarder << endl;
    }
};

int main(int argc, char** argv){
    string trainFilePath = "data/sample.txt";
    string testFilePath = "data/sample.txt";
    if (argc == 3){
        trainFilePath = string(argv[1]);
        testFilePath = string(argv[2]);
    }
    
    // データセット管理用の変数（すべてインデックスIDで管理）
    map<string, int> wordToId;
    vector<vector<TYPE>> wordVectors;
    map<string, int> labelToId;
    vector<vector<TYPE>> labelVectors;
    vector<string> idToLabel;
    vector<pair<vector<int>, int>> trainDataset;
    
    // データ読み込みと変換
    createDataset(trainFilePath, trainDataset, wordToId, wordVectors, labelToId, labelVectors, idToLabel);

    Perceptron ppn = Perceptron(30, (TYPE)0.1, idToLabel.size());
    
    // 時間計測開始
    auto start_train = chrono::high_resolution_clock::now();
    
    ppn.fit(trainDataset, wordVectors, labelVectors);
    
    auto end_train = chrono::high_resolution_clock::now();
    auto train_time = chrono::duration_cast<chrono::milliseconds>(end_train - start_train).count();
    cout << "finished training" << endl;
    
    auto start_test = chrono::high_resolution_clock::now();
    
    ppn.test(testFilePath, wordToId, wordVectors, labelToId, idToLabel);

    auto end_test = chrono::high_resolution_clock::now();
    auto test_time = chrono::duration_cast<chrono::milliseconds>(end_test - start_test).count();

    cout << "学習時間: " << train_time << " ms" << endl;
    cout << "テスト時間: " << test_time << " ms" << endl;

    return 0;
}