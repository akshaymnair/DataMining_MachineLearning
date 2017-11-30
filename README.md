# Phase3

Python version 2.7

```bash
cd Code/
pip install pandas scipy sklearn
```

Task 1 & 2 instruction:
python task1/pca.py 3189 3216 3233
python task1/svd.py 3189 3216 3233
python task1/lda.py 3189 3216 3233
python task1/tensordecomposition.py 3189 3216 3233
python task1/pagerank.py 3189 3216 3233
python task1/combined.py 3189 3216 3233

Task 3 Instructions
Run python map.py
This reduces dimensions of movies to 500 and stores in file movie_matrix_svd.pkl

Run python lsh.py #layers #hashes #movie_list
 #movie_list is optional(all movies considered when given empty)
Eg:- python lsh.py 5 10 
     python lsh.py 5,10 65,3111
The above code asks questions on console
Sample questions on console
Enter movie id to query:- 12
Enter range of query:- 1 

Task 4 Instructions
Interface for task4 is provided with task3 code itself
Sample quesitons on console
Give Feedback as String of 1(Good),0(Nuetral),-1(Bad) :- 1
66    0.835438
Name: CosineValues, dtype: float64
Give Feedback as String of 1(Good),0(Nuetral),-1(Bad) :- 

Task 5 Instructions
Input file is movie_labels.csv
Run python rnn.py
It outputs rest of movies and classified labels
