import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 1000)

DATA_PATH = "/Users/pcpu/Desktop/job_search/JPMC_Asset and Wealth Management Technology/assessment"
FILE_NAME = "enron_test.csv"
NULL_VALUE = "NA"
TOPIC_COUNT = 5


def convert_column_to_date(df, column_name=None, columns=None):
    """Converts column or list of columns to date format"""
    if isinstance(column_name, str):
        df[column_name] = pd.to_datetime(df[column_name], infer_datetime_format=True, utc=True)
    elif isinstance(columns, list):
        for column in columns:
            df[column] = pd.to_datetime(df[column], infer_datetime_format=True, utc=True)
    else:
        print("Specify either column_name='example_name' or columns=[<column name list>]")
    return df


def extract_sender_receiver_name(row):
    """Gets name of sender/receiver for email"""
    if row == NULL_VALUE:
        return NULL_VALUE
    else:
        return re.sub("\'", "", re.findall("{(.*)}", row)[0])


def get_email_count_by_name(df, type=None):
    """Gets email counts by name"""
    counts_df = df \
        .groupby(type)[type] \
        .count() \
        .to_frame() \
        .rename(columns={type: "count"}) \
        .reset_index() \
        .sort_values("count", ascending=False) \
        .reset_index(drop=True)
    return counts_df


def get_unique_emails_from_list(row):
    """Gets the unique list of emails from list"""
    if len(row) > 0:
        return set([re.sub(r"<|>", "", email) for email in re.findall("<(.*)>", row[0])])
    else:
        return row


def main():
    # read data into data frame
    data = os.path.join(DATA_PATH, FILE_NAME)

    df = pd.read_csv(data)
    df.columns = [column.lower() for column in df.columns]

    # add new features
    df = convert_column_to_date(df, columns=["date", "new_date"])
    df["year"], df["month"], df["day"] = df.date.dt.year, df.date.dt.month, df.date.dt.day

    # get counts by year and month
    df.groupby(["year", "month"])["year"].count()

    df["to"].fillna(NULL_VALUE, inplace=True)

    for column in "to", "from":
        df[column] = df[column].apply(lambda row: extract_sender_receiver_name(row))

    # get email counts and show top counts
    email_counts = {type: get_email_count_by_name(df, type=type) for type in ["to", "from"]}
    print("\nCounts for most common email receiver:")
    print(email_counts["to"].loc[:20, :])

    print("\nCounts for most common email sender:")
    print(email_counts["from"].loc[:20, :])

    forward_cond = lambda row: 1 if re.search("forwarded by", row.lower()) else 0
    df = df.assign(forwarded=df["content"].apply(forward_cond))

    # show emails with no subject
    for i in range(10):
        print(i)
        content = list(df[df["subject"].isnull()]["content"])
        print(content[i])

    # transform emails to get features for model
    featurizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    analyze = featurizer.build_analyzer()
    content = df["content"].apply(lambda row: analyze(row))

    # show most common words across emails
    word_counts = pd.Series(np.concatenate(content)).value_counts()
    print(word_counts[:10])

    featurizer = featurizer.fit(df["content"])

    # get vocab and show most common words
    vocab = featurizer.vocabulary_
    index_to_term = {index: term for term, index in vocab.items()}

    word_counts = featurizer.transform(df["content"])
    X = word_counts
    y = df["subject"].isnull()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("\nPerformance of classification model:")
    print(classification_report(y_test, pred))

    feature_importances = clf.feature_importances_
    feature_ranking = list(zip(range(len(feature_importances)), feature_importances))
    feature_ranking = sorted(feature_ranking, key=lambda row: -row[1])

    print("\nFeature importance ranking:")
    print(feature_ranking)

    term_to_importance = [(index_to_term.get(id), importance) for id, importance in feature_ranking]

    for rank, term in enumerate(term_to_importance[:10], start=1):
        print(str(rank) + ": " + term[0])

    lda = LatentDirichletAllocation(n_components=TOPIC_COUNT, random_state=0)
    model = lda.fit(X)

    top_words = {}
    for topic in range(TOPIC_COUNT):
        top_words_idx = model.components_[topic].argsort()[:-10:-1]
        top_words_list = [index_to_term.get(idx) for idx in top_words_idx]
        top_words[topic] = top_words_list

    top_words_df = pd.DataFrame(top_words)
    top_words_df.columns = ["topic" + str(id + 1) for id in top_words_df.columns]

    emails_with_recipient_info = df[df["content"].apply(lambda row: row.find("To:") != -1)]

    included = lambda row: re.findall(r"To:(.*)cc:", row.replace("\n", " "))
    emails_with_recipient_info["info"] = emails_with_recipient_info["content"].apply(included)

    emails_with_recipient_info["unique_emails"] = \
        emails_with_recipient_info["info"].apply(get_unique_emails_from_list)


if __name__ == "__main__":
    main()
