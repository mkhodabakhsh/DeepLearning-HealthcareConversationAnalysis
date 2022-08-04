# DeepLearning-HealthcareConversationAnalysis

This project focuses on detecting open/closed-ended questions, affirmation statements, and reflective statements from healthcare conversation transcripts using a hybrid rule-based and machine learning approach.

## Project Overview

The project implements a hybrid methodology combining rule-based and deep learning models to extract key features from healthcare conversations. Key features include identifying:
- Open-ended and closed-ended questions
- Affirmation statements
- Reflective statements

### Methodology
1. Text cleaning and preprocessing.
2. Sentence detection using the NLTK library.
3. Detection of question sentences using a DL model.
4. Rule-based classification of questions into open-ended and closed-ended.
5. Use of a probabilistic RNN model for token-wise dialogue act feature probability.
6. Correlation calculation between dialogue act features and affirmation/reflection statements.
7. Prediction of affirmation and reflection using a fused machine learning model.
