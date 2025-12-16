from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util


class SBERTScore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def compute(self, predictions, references):
        pred_texts = [p["prediction_text"] for p in predictions]
        ref_texts = [r["answers"]["text"][0] for r in references]

        emb_preds = self.model.encode(
            pred_texts, convert_to_tensor=True, show_progress_bar=False
        )
        emb_refs = self.model.encode(
            ref_texts, convert_to_tensor=True, show_progress_bar=False
        )

        cosine_scores = st_util.pairwise_cos_sim(emb_preds, emb_refs).tolist()

        return cosine_scores
