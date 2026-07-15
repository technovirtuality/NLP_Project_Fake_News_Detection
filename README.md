# Fake_News_Detection

Objectives 

1.To design and develop a machine learning model that can accurately classify news articles as real or fake.

2. To build a model which will identify and extract key features (linguistic, semantic or contextual) that differentiate fake news from real news.

3. To evaluate and compare the performance of different machine learning algorithms like Logistic Regression, Naïve Baye’s, Random Forest, Gradient Boosting, Neural Networks for fake news detection.

4. To investigate the role of natural language processing (NLP) techniques in improving the accuracy of fake news detection.



Dataset : https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
The dataset had 16 columns namely index, the ID of the statement([ID].json), label, statement, subject, speaker, speaker’s job title,, the state info, party affiliation, the total  credit history account, including current statement(comprises of 5 columns together which are barely true counts, false counts, half true counts, mostly true counts, pants on fire counts), context(venue /location of speech statement) and extracted justification.  These include data on subject, speaker, state and party affiliations. No purchase or license is needed for these datasets, its ideal for academic and research. There are a total of 12788 rows. There are 10239 rows for testing,1266 rows for testing and 1283 rows for validation.


<img width="915" height="427" alt="image" src="https://github.com/user-attachments/assets/d97ee937-ed28-4c35-a888-9b21a7ba590a" />

Data Cleaning:
Check data imbalance, Combining train-validation data, Merging statement with justification column and Removing rows with null values from the dataset. This step checks if there are missing values in the target variable 'label'. There are 2 missing values in ‘label’ dataset. We have dropped some columns as well which are index, id of the statement, state info, context, justification, barely true counts, false counts, half true counts, mostly true counts, pants on fire counts. The training and validation data are combined into one dataset. The columns ‘statement’ and ‘justification’ were merged together to form a new column named “new”. The models are trained on his ‘new; column of training dataset. Next, the rows with nan values are removed from the dataset in order to reduce noise from the data. 

Feature Extraction: There are two methods of pre-processing adopted for model building: 
Pre-processing1: CountVectorizer + TfidfVectorizer, pre-processing2: TF-IDF (Count vectorizer + TFIDF) with n-gram. The first method is performed by building a Bag-of-Words model first with CounterVectorizer combined to TfidfVectorizer. The CountVectorizer method is used so that text data can be converted into word-count-vectors. This process gives an easy method to both tokenize a aggregation of text documents and to build a vocabulary of known words. Also, the vocanulary is utilized to encode new documents.


Six way classification: 
For each classification model, a machine learning pipeline was implemented to combine the preprocessing stage (text vectorization) with the classifier. The models were trained using the 'new' column of the training dataset, while predictions were generated for the 'statement' column of the test dataset. Model performance was evaluated using 5-fold cross-validation, with the F1-score and confusion matrix employed as the primary evaluation metrics.

The multi-class classification framework was developed using three different feature extraction approaches:

First Approach: CountVectorizer + TF-IDF Transformation
In the first approach, textual data were initially transformed into word frequency counts using CountVectorizer. These frequency counts were subsequently converted into Term Frequency–Inverse Document Frequency (TF-IDF) representations using TfidfTransformer, thereby assigning greater importance to informative words while reducing the influence of frequently occurring terms. The extracted TF-IDF features were then used to train a Logistic Regression classifier within a pipeline. The trained model was evaluated on the test dataset using accuracy and other standard classification performance metrics.

Second Approach: TF-IDF Vectorizer with N-grams
The second approach directly transformed raw text into TF-IDF-weighted feature vectors using TfidfVectorizer. During preprocessing, common English stop words (such as the, is, and and) were removed to eliminate less informative terms. In addition, both unigrams (single words) and bigrams (two consecutive words) were extracted as features, enabling the model to capture meaningful phrases such as "fake news" and "not true" that provide richer contextual information than individual words alone. After fitting the vectorizer on the training data, the vocabulary and corresponding IDF weights were learned, and the text was transformed into a sparse TF-IDF feature matrix. These features were then used to train a Logistic Regression classifier, which was subsequently evaluated on the test dataset to measure its classification performance.

Third Approach: CountVectorizer Only
The third approach employed CountVectorizer as the sole feature extraction technique. In this method, textual data were represented based solely on the frequency of word occurrences within each statement. Unlike TF-IDF, this representation does not assign higher weights to distinctive words or reduce the importance of commonly occurring words. Consequently, it provides a simpler and more straightforward feature extraction mechanism. The resulting count-based feature vectors were then used to train the classification models, and their performance was evaluated using the same experimental framework as the other approaches.

<img width="851" height="387" alt="image" src="https://github.com/user-attachments/assets/7be457e3-a517-4a93-92dd-89937b6def4e" />

Binary Classification:

To perform binary classification, the original multi-class dataset was transformed by retaining only the true and false statement categories while excluding all other classes. In this binary setting, true statements were labeled as real (1), whereas false statements were labeled as fake (0). The same preprocessing steps used in the multi-class classification task, including data cleaning, data merging, and feature extraction, were applied to the binary dataset.

Feature extraction was carried out using two different approaches. The first approach employed CountVectorizer to convert textual data into word frequency vectors. The second approach combined CountVectorizer with TF-IDF transformation, where the generated word count vectors were converted into TF-IDF representations to assign greater importance to informative terms while reducing the influence of commonly occurring words.

The binary classification models were trained and evaluated using the same machine learning algorithms employed in the six-way classification task, namely Logistic Regression (LR), Support Vector Machine (SVM), Naïve Bayes (NB), Random Forest (RF), and Extreme Gradient Boosting (XGBoost). For each algorithm, models were developed using both count-based and TF-IDF feature representations to enable a comparative analysis of their performance.

Model performance was assessed using accuracy, F1-score, and the confusion matrix, providing a comprehensive evaluation of the classification results. In addition, feature importance analysis was performed for the XGBoost model to identify the most influential words and n-grams contributing to the prediction process. This analysis improves the interpretability of the model by highlighting the textual features that have the greatest impact on distinguishing between real and fake statements.

Results:
1. XGBoost:
   <img width="742" height="280" alt="image" src="https://github.com/user-attachments/assets/50f4be61-3a5a-48ee-ae54-bfbb33ccdbe4" />

2. Naive Bayes:
   <img width="732" height="287" alt="image" src="https://github.com/user-attachments/assets/1dccafaa-4c93-49c2-9c63-e3dc1203f861" />



Conclusion:
The performance of four machine learning models—XGBoost, Random Forest, Naïve Bayes, and Logistic Regression—was evaluated on the LIAR dataset for binary fake news detection. The experimental results reveal the significant impact of class imbalance and emphasize the importance of developing models capable of identifying the minority class rather than simply predicting the majority class.
Among the evaluated models, XGBoost achieved the best overall performance. It attained a precision of 0.28 for the real news class (Class 1) and was able to correctly classify 5 out of 208 real news instances. Although the recall for Class 1 was only 0.02, it demonstrates that XGBoost possesses some ability to distinguish between real and fake news. The model achieved an overall accuracy of 83%; however, this high accuracy is primarily a consequence of the dominance of fake news samples (Class 0) in the dataset rather than effective classification of both classes.
The Random Forest classifier performed slightly worse than XGBoost. It correctly identified only 2 real news instances, resulting in a recall of 0.01 for Class 1. Despite its effectiveness in many classification problems, Random Forest exhibited a strong bias toward the majority class and struggled to recognize real news samples.
Both Naïve Bayes and Logistic Regression showed poor performance on the minority class. Each model classified every instance as fake news (Class 0), producing zero true positives for the real news class. Consequently, both precision and recall for Class 1 were 0.00, as reflected in their confusion matrices. Although these models achieved overall accuracies of approximately 83–84%, these values are misleading because they result solely from correctly classifying the majority class while completely failing to detect real news.
Overall, XGBoost outperformed the other classifiers, although its ability to identify real news remained limited. The findings indicate that the principal challenge lies in the severe class imbalance of the LIAR dataset, together with the complexity of linguistic patterns that differentiate fake and real news. The combination of high overall accuracy and extremely low recall for the minority class suggests that the models are heavily biased toward the majority class rather than learning robust decision boundaries.
For fake news detection, evaluation metrics such as precision, recall, and F1-score for the minority class provide a more meaningful assessment of model performance than overall accuracy alone. While XGBoost serves as the strongest baseline among the evaluated algorithms, its limited effectiveness demonstrates that additional techniques are required to improve minority class detection. Future work should focus on addressing class imbalance through methods such as SMOTE, random undersampling, or class-weighted loss functions, as well as incorporating more informative feature engineering and advanced deep learning approaches, including transformer-based language models. These improvements have the potential to enhance the detection of real news and produce a more reliable and practical fake news detection system.


