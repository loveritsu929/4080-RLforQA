from utility import *

begin_time = datetime.datetime.now()
word_vocabulary = [None]
word_embedding = [[0.0] * glove_size]
word_count = {}
word_vector = {}

for record in load_file(train_dataset_path, "jsn") + load_file(develop_dataset_path, "jsn"):
    for _, paragraph in record["context"]:
        for sentence in paragraph:
            for token in spacy_nlp(sentence):
                word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1

    for token in spacy_nlp(record["question"]):
        word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1

for item in load_file(glove_archive_path, "txt"):
    glove_elements = item.strip().split(" ")

    if glove_elements[0] in word_count:
        word_vector[glove_elements[0]] = [float(element) for element in glove_elements[1:glove_size + 1]]

for word in sorted(word_count, key=word_count.get, reverse=True):
    if word in word_vector:
        word_vocabulary.append(word)
        word_embedding.append(word_vector[word])

train_composite = convert_dataset(load_file(train_dataset_path, "jsn"), word_vocabulary, True)
develop_composite = convert_dataset(load_file(develop_dataset_path, "jsn"), word_vocabulary, False)
dump_data(word_vocabulary, word_vocabulary_path, "obj")
dump_data(word_embedding, word_embedding_path, "obj")
dump_data(train_composite, train_composite_path, "obj")
dump_data(develop_composite, develop_composite_path, "obj")
print("preprocess: cost {} seconds".format(int((datetime.datetime.now() - begin_time).total_seconds())))
