import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

from Settings import Settings
from StringUtils import StringUtils


def apply_lda_to_clusters(clusters, class_docs, num_topics=0):

    lda_result = []
    cluster_docs = []

    for cluster_id, classes in clusters.items():
        cluster_bag_of_words = []

        for class_id in classes:
            class_doc = class_docs[class_id]
            cluster_bag_of_words.extend(class_doc)

        cluster_docs.append((cluster_id, cluster_bag_of_words))

    if Settings.K_TOPICS:
        num_topics = Settings.K_TOPICS
        texts, corpus, dictionary = clear_documents(cluster_docs)
        # print information about preprocessed data
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))
        model = fit_lda(corpus, num_topics, dictionary)

        with open("topic_model_results.txt", "w") as f:

            for topic_id in range(model.num_topics):
                topic_terms = model.show_topic(topic_id)
                f.write(f"topic {topic_id}:\n")
                for term, prob in topic_terms:
                    f.write(f"{term}: {prob}\n")
                f.write("\n")

            f.write("\n")

            # topic infer
            for doc, bow in zip(texts, corpus):
                f.write(f"doc content: {doc}\n")
                for topic, prob in model.get_document_topics(bow):
                    f.write(f"Topic {topic}: {prob}\n")
                f.write("\n")

    else:
        lda_result, num_topics = find_best_lda(cluster_docs)


    print(123)

def clear_documents(docs):
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # with open(f"{Settings.DIRECTORY}/data/words/{Settings.PROJECT_NAME}_{Settings.ID}", 'w') as f:
    #     for d in docs:
    #         f.write(d + "\n")

    #     f.write("\n")

    # Clean text based on java stop words
    docs_content = []
    for doc in docs:
        directory = f"{Settings.DIRECTORY}/data/{Settings.PROJECT_NAME}/{doc[0]}"
        with open(directory, "w+") as f:
            f.write(f"{doc[0]}\n")
            f.write("Before processing:\n")
            f.write(f"{doc[1]}\n")

        doc_content = StringUtils.clear_text(doc[1])
        docs_content.append(doc_content)

        with open(directory, "a+") as f:
            f.write("\nAfter processing:\n")
            f.write(f"{doc_content}\n")

    # compile sample documents into a list
    doc_set = docs_content

    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for text in doc_set:
        # clean and tokenize document string
        raw = text.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [t for t in tokens if not t in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(st) for st in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # filter dictionary from outliers
    dictionary.filter_extremes(no_below=3, no_above=0.75, keep_n=1000)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    return texts, corpus, dictionary

def fit_lda(corpus, num_topics, dictionary):
    lm = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    return lm

def find_best_lda(docs):
    len_docs = len(docs)

    step = 1
    if len_docs < 50:
        start = 4
        end = 12
    elif len_docs < 100:
        start = 4
        end = 16
        step = 1
    elif len_docs < 200:
        start = 10
        end = 25
        step = 3
    elif len_docs < 500:
        start = 12
        end = 30
        step = 3
    elif len_docs < 1000:
        start = 15
        end = 35
        step = 3
    else:
        start = 14
        end = 40
        step = 3

    texts, corpus, dictionary = clear_documents(docs)

    model_list, coherence_values, num_topics = compute_coherence_values(
        dictionary=dictionary, corpus=corpus, texts=texts, start=start, limit=end, step=step)

    x = range(start, end, step)
    for model, coherence, k in zip(model_list, coherence_values, num_topics):
        print(f"k {k} - coherence {coherence} lda_model {model}")

    # plt.plot(x, coherence_values)
    # plt.xlabel('number of topics')
    # plt.ylabel('topic coherence')
    # plt.show()

    S = 5
    best_topic = None
    while best_topic == None and S > 0:
        knee = KneeLocator(x, coherence_values, curve='concave',
                           direction='increasing', S=S)

        # Plot knee of coherence over number of topics
        # knee.plot_knee()
        # plt.xlabel('number of topics')
        # plt.ylabel('topic coherence')
        # plt.show()

        best_topic = knee.knee

        S -= 1
        print(f"Trying knee of S={S}")

    # In case the knee isn't found, select the max. Happens when the coherence values are very similar across topics
    if best_topic == None:
        coherence_list = list(coherence_values)
        best_topic = x[coherence_list.index(max(coherence_list))]
    Settings.K_TOPICS = best_topic
    print(
        f"The knee of topics/coherence is {best_topic}")

    lda_model = None
    for model, k_topic in zip(model_list, num_topics):
        if k_topic == best_topic:
            lda_model = model
            break
    topics_per_doc = []

    # topics_per_doc = [lda_model.get_document_topics(corp) for corp in corpus]

    for c in lda_model[corpus]:
        topics_per_doc.append(c)

    # print(f"Topics per doc: {topics_per_doc}")
    # print(f"Topics: {lda_model.show_topics()}")

    return topics_per_doc, best_topic