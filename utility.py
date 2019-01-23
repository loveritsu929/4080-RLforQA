import datetime, json, os, pickle, spacy
from tqdm import tqdm

glove_size = 300
train_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/train_dataset")
develop_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/develop_dataset")
glove_archive_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/glove_archive")
word_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/word_vocabulary")
word_embedding_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/word_embedding")
train_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/train_composite")
develop_composite_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/media/data1/hotpot/develop_composite")
spacy_nlp = spacy.load(name="en_core_web_lg", disable=["tagger", "parser", "ner"])

def load_file(file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return file_stream.read().splitlines()

    elif file_type == "jsn":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return json.load(file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="rb") as file_stream:
            return pickle.load(file_stream)

    else:
        pass


def dump_data(data_buffer, file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            file_stream.write("\n".join(data_buffer))

    elif file_type == "jsn":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            json.dump(obj=data_buffer, fp=file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="wb") as file_stream:
            pickle.dump(obj=data_buffer, file=file_stream)

    else:
        pass


def convert_dataset(dataset_buffer, word_vocabulary, require_answer):
    def get_text_symbols(text_tokens):
        return [token.text for token in text_tokens]

    def get_text_numbers(text_tokens):
        return [word_vocabulary.index(token.text) if token.text in word_vocabulary else 0 for token in text_tokens]

    def get_text_span(text_tokens, span_range):
        for start_index, start_token in enumerate(text_tokens):
            if span_range[0] == start_token.idx:
                for end_index, end_token in enumerate(iterable=text_tokens[start_index:], start=start_index):
                    if span_range[1] == end_token.idx + len(end_token.text):
                        return [start_index, end_index]

    composite_records = []

    for record in dataset_buffer:
        fact_handles = []
        sentence_source_array = []
        sentence_tokens_array = []
        sentence_length_array = []
        sentence_symbols_array = []
        sentence_numbers_array = []

        for title, paragraph in record["context"]:
            for index, sentence in enumerate(paragraph):
                fact_handles.append([title, index])
                sentence_source_array.append(sentence.strip())
                sentence_tokens_array.append(spacy_nlp(sentence_source_array[-1]))
                sentence_length_array.append(len(sentence_tokens_array[-1]))
                sentence_symbols_array.append(get_text_symbols(sentence_tokens_array[-1]))
                sentence_numbers_array.append(get_text_numbers(sentence_tokens_array[-1]))

        sentence_source_array.append(record["question"].strip())
        sentence_tokens_array.append(spacy_nlp(sentence_source_array[-1]))
        sentence_length_array.append(len(sentence_tokens_array[-1]))
        sentence_symbols_array.append(get_text_symbols(sentence_tokens_array[-1]))
        sentence_numbers_array.append(get_text_numbers(sentence_tokens_array[-1]))

        sentence_symbols_array = [
            symbols + [""] * (max(sentence_length_array) - len(symbols))
            for symbols in sentence_symbols_array
        ]

        sentence_numbers_array = [
            numbers + [0] * (max(sentence_length_array) - len(numbers))
            for numbers in sentence_numbers_array
        ]

        composite_record = {
        	
        	"id": record["_id"],
            "fact_handles": fact_handles,
            "sentence_source_array": sentence_source_array,
            "sentence_length_array": sentence_length_array,
            "sentence_symbols_array": sentence_symbols_array,
            "sentence_numbers_array": sentence_numbers_array
        }

        if require_answer:
            fact_labels = [1 if handle in record["supporting_facts"] else 0 for handle in fact_handles]
            answer_class = 0 if record["answer"].strip() == "no" else 1 if record["answer"].strip() == "yes" else 2
            spfact_tokens_array = [tokens for tokens, label in zip(sentence_tokens_array, fact_labels) if label == 1]

            if spfact_tokens_array:
                answer_spans = [[0, sum(map(len, spfact_tokens_array)) - 1]] if answer_class != 2 else [
                    [offset + span[0], offset + span[1]]
                    for offset, span in [
                        (
                            sum(map(len, spfact_tokens_array[:spfact_index])),
                            get_text_span(spfact_tokens, [char_index, char_index + len(record["answer"].strip())])
                        )
                        for spfact_index, spfact_tokens in enumerate(spfact_tokens_array)
                        for char_index in range(len(spfact_tokens.text))
                        if spfact_tokens.text[char_index:].startswith(record["answer"].strip())
                    ]
                    if span
                ]

                if answer_spans:
                    composite_record["fact_labels"] = fact_labels
                    composite_record["answer_class"] = answer_class
                    composite_record["answer_spans"] = answer_spans
                    composite_records.append(composite_record)

        else:
            composite_records.append(composite_record)

    return composite_records
