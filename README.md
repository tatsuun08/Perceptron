# Perceptron実装
マクロによるデータ型の切り替え USE_FLOAT

## 実行方法
### **コンパイル** <br>
例）Windows GCC <br>
```
gcc main.cpp -o main.exe
```
### **実行**
```
main.exe train_file.txt test_file.txt
```

コンパイルオプション
```
単精度浮動小数（Float）
gcc main.cpp -o main.exe -DUSE_FLOAT
倍精度浮動小数（Double）
gcc main.cpp -o main.exe
```

## データの処理
```
連想配列 String word -> Int wordId <br>
連想配列 String label -> Int LabelId <br>

String -> hash_seed -> randomVector(hash_seed)

Vector<TYPE> wordVectors[wordId]
Dataset {Vector<TYPE> ArticleWords(wordVectors1..wordVectorsN), Int label}[i] (0..N)
```