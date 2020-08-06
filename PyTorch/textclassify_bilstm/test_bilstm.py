from textclassifier import TextClassifier

num_repeat = 3

for i in range(num_repeat):
    classifier = TextClassifier()
    classifier.train()
    classifier.test()
