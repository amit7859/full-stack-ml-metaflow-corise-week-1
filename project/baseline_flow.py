from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] >= 4 else 0


class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        #### ADDED  
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            analyzer = 'word',
            lowercase = True,
        )
        vectorized_reviews = vectorizer.fit_transform(reviews)

        ####


        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})

        self.traindf, self.valdf, self.train_vectors, self.val_vectors = train_test_split(_df, vectorized_reviews, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")
        print(f"num of rows in train set review vectors: {self.train_vectors.shape[0]}")
        print(f"num of rows in validation set: {self.val_vectors.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"

        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.

        from sklearn.linear_model import LogisticRegression
        log_model = LogisticRegression()

        log_model = log_model.fit(X=self.train_vectors, y=self.traindf['label'])
        y_pred = log_model.predict(self.val_vectors)

        from sklearn.metrics import accuracy_score, roc_auc_score

        self.base_acc = accuracy_score(self.valdf['label'].to_numpy(), y_pred)
        self.base_rocauc = roc_auc_score(self.valdf['label'].to_numpy(), y_pred)

        self.valdf["prediction"] = y_pred 
        self.valdf = self.valdf

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        # self.valdf.head()
        
        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        # False Positives

        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.
        false_positives_df = self.valdf[(self.valdf["label"] == 0) & (self.valdf["prediction"] == 1)]

        current.card.append(Markdown("## Number of False Positives"))
        current.card.append(Artifact(len(false_positives_df)))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table

        from metaflow.cards import Table
        current.card.append(
            Table.from_dataframe(
                false_positives_df.head(20) 
            )
        )

        # False Negatives

        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1.
        false_negatives_df = self.valdf[(self.valdf["label"] == 1) & (self.valdf["prediction"] == 0)]

        current.card.append(Markdown("## Number of False Positives"))
        current.card.append(Artifact(len(false_negatives_df)))

        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: display the false_negatives dataframe using metaflow.cards
        from metaflow.cards import Table
        current.card.append(
            Table.from_dataframe(
                false_negatives_df.head(20) 
            )
        )

if __name__ == "__main__":
    BaselineNLPFlow()
