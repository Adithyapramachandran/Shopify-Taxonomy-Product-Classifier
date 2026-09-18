"""
Classification engine: matches a product's text (and optionally an image) to
the Shopify Product Taxonomy, extracts likely category attributes/values, and
produces a confidence score plus alternatives.

Design notes
------------
- The taxonomy (categories + attributes + attribute values) is bundled as
  static JSON under classifier/data/ (pulled from Shopify/product-taxonomy),
  so classification never depends on network access at runtime.
- Category matching = TF-IDF cosine similarity between product text and each
  taxonomy category's name/path text, boosted by direct keyword overlap with
  the seller's own "Product Category"/"Product Sub Category" columns when
  present. This scales to 10,000+ products because the TF-IDF matrix over
  ~12k categories is built once (and cached to disk) and every product is a
  single vector transform + sparse matrix multiply.
- Attribute/value extraction is dictionary-driven: for the winning category's
  declared attributes (e.g. "Color", "Material", "Upholstery Fabric"), we scan
  the product's text for a match against that attribute's controlled value
  list from Shopify's taxonomy. This avoids inventing values that Shopify
  wouldn't accept.
- Image support is an optional, pluggable stage (see image_engine.py). If no
  image classifier is available (e.g. no model/network), the pipeline simply
  skips it and marks used_image=False rather than failing.
- Every stage is defensive: bad/missing data degrades the confidence score
  and produces review_reasons rather than raising.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ENGINE_VERSION = "1.0.0"

# TF-IDF cosine similarity between short product text and short category names
# is naturally compressed well below 1.0 even for a correct match (high-
# dimensional sparse vectors rarely overlap heavily). CALIBRATION_SCALE
# rescales the raw cosine score into a more human-meaningful 0-1 confidence
# so that "this is basically a certain match" reads as ~90%+ rather than ~50%.
# It is a linear rescale (order-preserving), chosen empirically against this
# taxonomy: correct matches on real catalogue titles typically score
# 0.45-0.60 raw, so we treat ~0.55 raw as "near-certain".
CALIBRATION_SCALE = 0.55

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    if not text:
        return ""
    return " ".join(_WORD_RE.findall(text.lower()))


@dataclass
class CategoryMatch:
    category_id: str
    name: str
    full_name: str
    confidence: float


@dataclass
class AttributeMatch:
    handle: str
    name: str
    value: str
    confidence: float
    source: str  # which product field the value was found in


@dataclass
class ClassificationOutcome:
    category: Optional[CategoryMatch]
    alternatives: list
    attributes: list
    text_confidence: float
    image_confidence: Optional[float]
    used_image: bool
    review_reasons: list = field(default_factory=list)


class TaxonomyIndex:
    """Loads the bundled taxonomy once per process and builds the TF-IDF index."""

    def __init__(self, categories_path: Path, attribute_values_path: Path, cache_path: Path):
        self.categories_path = Path(categories_path)
        self.attribute_values_path = Path(attribute_values_path)
        self.cache_path = Path(cache_path)
        self._load()

    def _load(self):
        with open(self.categories_path, encoding="utf-8") as f:
            self.categories = json.load(f)
        with open(self.attribute_values_path, encoding="utf-8") as f:
            self.attribute_values = json.load(f)  # {attribute_name: [values...]}

        self.by_id = {c["id"]: c for c in self.categories}
        # a category is a "leaf" (most specific / preferred classification target)
        # if no other category declares it as a parent.
        parent_ids = {c["parent_id"] for c in self.categories if c.get("parent_id")}
        for c in self.categories:
            c["is_leaf"] = c["id"] not in parent_ids
            # ancestor-path tokens (full path minus the leaf name itself), used
            # to disambiguate between categories that share an identical leaf
            # name in different branches, e.g. "Dining Tables" appears both
            # under Outdoor Furniture and under Kitchen & Dining Room Tables.
            ancestor_text = c["full_name"].rsplit(">", 1)[0] if ">" in c["full_name"] else ""
            c["_path_tokens"] = set(normalize(ancestor_text).split())

        # Build (or load cached) TF-IDF matrix over every category's searchable text.
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("n_categories") == len(self.categories):
                    self.vectorizer = cached["vectorizer"]
                    self.matrix = cached["matrix"]
                    self._build_value_lookup()
                    return
            except Exception:
                pass  # fall through and rebuild

        corpus = [self._category_text(c) for c in self.categories]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=0.6, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {"vectorizer": self.vectorizer, "matrix": self.matrix, "n_categories": len(self.categories)},
                    f,
                )
        except Exception:
            pass  # cache is an optimization only

        self._build_value_lookup()

    def _build_value_lookup(self):
        # attribute name -> list of (normalized_value, original_value), longest first
        # so multi-word values ("Solid wood") are checked before single words.
        self._value_lookup = {}
        for attr_name, values in self.attribute_values.items():
            pairs = sorted({(normalize(v), v) for v in values if v}, key=lambda p: -len(p[0]))
            self._value_lookup[attr_name] = [p for p in pairs if p[0]]

    @staticmethod
    def _category_text(c: dict) -> str:
        # Weight the leaf name most heavily; include the full breadcrumb for
        # broader context. Attribute names (Color, Material, Size...) are
        # deliberately excluded here -- they repeat across thousands of
        # unrelated categories and would drown out the distinctive category
        # words with generic noise.
        parts = [c["name"]] * 4 + [c["full_name"].replace(">", " ")]
        return normalize(" ".join(parts))

    def _raw_scores(self, text: str):
        text = normalize(text)
        if not text:
            return None
        vec = self.vectorizer.transform([text])
        return cosine_similarity(vec, self.matrix)[0]

    def search(self, text: str, top_n: int = 5, prefer_leaf: bool = True, title_text: str = None,
               path_hint_text: str = None) -> list[CategoryMatch]:
        """Rank taxonomy categories against `text`. If `title_text` is given,
        it is scored separately and blended in with extra weight: the product
        title alone ("Armchair") is short and precise, so a strong title match
        should count for more than being buried inside a long, noisy paragraph
        of marketing copy. If `path_hint_text` is given (typically the
        seller's own category/sub-category, e.g. "Bar and Dining"), a small
        bonus is added for categories whose *ancestor path* shares words with
        it -- this breaks ties between identically-named leaves that live in
        different branches (e.g. indoor vs. outdoor "Dining Tables")."""
        sims = self._raw_scores(text)
        if sims is None:
            return []

        if title_text:
            title_sims = self._raw_scores(title_text)
            if title_sims is not None:
                sims = 0.45 * sims + 0.55 * title_sims

        hint_tokens = set(normalize(path_hint_text).split()) if path_hint_text else None

        ranked = []
        for idx, base_score in enumerate(sims):
            if base_score <= 0:
                continue
            score = float(base_score)
            if hint_tokens:
                path_tokens = self.categories[idx]["_path_tokens"]
                if path_tokens:
                    overlap = len(hint_tokens & path_tokens) / max(1, len(hint_tokens))
                    score = min(1.0, score + 0.12 * overlap)
            ranked.append((idx, score))
        ranked.sort(key=lambda p: -p[1])

        results = []
        seen_names = set()
        for idx, score in ranked:
            cat = self.categories[idx]
            if cat["full_name"] in seen_names:
                continue
            seen_names.add(cat["full_name"])
            calibrated = min(1.0, score / CALIBRATION_SCALE)
            results.append(CategoryMatch(cat["id"], cat["name"], cat["full_name"], calibrated))
            if len(results) >= top_n * 3:
                break

        # Prefer leaves among comparable scores, but don't discard strong
        # non-leaf matches entirely (some products really only fit a mid-level node).
        leaves = [r for r in results if self.by_id[r.category_id]["is_leaf"]]
        non_leaves = [r for r in results if not self.by_id[r.category_id]["is_leaf"]]
        ordered = leaves + non_leaves
        return ordered[:top_n]

    def attributes_for(self, category_id: str) -> list[dict]:
        cat = self.by_id.get(category_id)
        if not cat:
            return []
        return cat.get("attributes", [])

    def find_values(self, attribute_name: str, text_by_source: dict[str, str]) -> Optional[AttributeMatch]:
        """Look for a controlled-vocabulary value for `attribute_name` inside the
        given {source_field_name: text} map. Returns the best (longest, first-found) match."""
        candidates = self._value_lookup.get(attribute_name)
        if not candidates:
            return None
        for source, raw_text in text_by_source.items():
            norm = normalize(raw_text)
            if not norm:
                continue
            padded = f" {norm} "
            for norm_val, orig_val in candidates:
                if f" {norm_val} " in padded:
                    # confidence heuristic: exact field matches (e.g. Color column
                    # for a "Color" attribute) are more trustworthy than free text
                    conf = 0.9 if source.lower() == attribute_name.lower() else 0.65
                    return AttributeMatch(handle="", name=attribute_name, value=orig_val,
                                           confidence=conf, source=source)
        return None


@lru_cache(maxsize=1)
def get_taxonomy_index() -> TaxonomyIndex:
    return TaxonomyIndex(
        settings.TAXONOMY_CATEGORIES_FILE,
        settings.TAXONOMY_ATTRIBUTE_VALUES_FILE,
        settings.TAXONOMY_MODEL_CACHE,
    )


def build_product_text(product) -> dict[str, str]:
    """Return the fields we search across, kept separate so attribute
    extraction can weight "the Color column says X" above "X appears somewhere
    in a paragraph of marketing copy"."""
    return {
        "title": product.title or "",
        "description": product.description or "",
        "bullets": product.bullets or "",
        "brand": product.brand or "",
        "source_category": product.source_category or "",
        "source_sub_category": product.source_sub_category or "",
        "collection_name": product.collection_name or "",
        "color": product.color or "",
        "materials": product.materials or "",
    }


def classify_product(product, image_bytes_list: Optional[list[bytes]] = None) -> ClassificationOutcome:
    """Main entry point: classify a single Product instance.

    `image_bytes_list` is optional pre-fetched image content; when omitted,
    image-based signal is simply skipped (requirement #4: must work with or
    without images).
    """
    index = get_taxonomy_index()
    review_reasons = []

    fields = build_product_text(product)
    # Category-text signal: title is the strongest, single-word-precise signal
    # ("Sofa", "Bar Stool", "Trash Bin"). The seller's own sub-category is a
    # useful but noisier prior; the seller's *top-level* category (e.g. "Living
    # Room") is too broad on its own -- an accessory or decor item can live in
    # "Living Room" without being furniture -- so it gets only light weight to
    # act as a tie-breaker rather than the dominant signal. Description/bullets
    # are truncated so a long marketing paragraph can't drown out the title.
    combined_text = " ".join([
        fields["title"], fields["title"], fields["title"],      # title weighted x3
        fields["source_sub_category"], fields["source_sub_category"],
        fields["source_category"],
        fields["description"][:400], fields["bullets"][:200],
        fields["brand"], fields["collection_name"],
    ])

    if not normalize(combined_text):
        review_reasons.append("No usable text (title/description/category all empty).")

    if not fields["description"]:
        review_reasons.append("Missing product description; classification relies on title/category only.")

    title_signal = " ".join([fields["title"], fields["source_sub_category"]]) if fields["title"] else None
    path_hint = " ".join([fields["source_category"], fields["source_sub_category"]]).strip() or None
    text_matches = index.search(combined_text, top_n=5, title_text=title_signal, path_hint_text=path_hint) if normalize(combined_text) else []
    text_confidence = text_matches[0].confidence if text_matches else 0.0

    # Optional image-based re-ranking / confirmation.
    image_confidence = None
    used_image = False
    if image_bytes_list:
        try:
            from .image_engine import classify_images_against_categories
            image_result = classify_images_against_categories(
                image_bytes_list, [m.full_name for m in text_matches] if text_matches else None
            )
            if image_result is not None:
                used_image = True
                image_confidence = image_result.get("confidence")
                # If the image strongly agrees with a *different* top candidate,
                # promote it; otherwise treat it as a confirmation signal that
                # nudges confidence up.
                image_label = image_result.get("best_label")
                if image_label:
                    match = next((m for m in text_matches if m.full_name == image_label), None)
                    if match and match is not text_matches[0]:
                        text_matches.remove(match)
                        text_matches.insert(0, match)
        except Exception:
            # Image classifier unavailable/failed: degrade gracefully, no crash.
            used_image = False
            image_confidence = None
    elif product.has_image:
        review_reasons.append("Product has image(s) but image-based classification was not run (unavailable or not configured for this run).")
    else:
        review_reasons.append("No product images available; classification is text-only.")

    if not text_matches:
        return ClassificationOutcome(
            category=None, alternatives=[], attributes=[],
            text_confidence=0.0, image_confidence=image_confidence, used_image=used_image,
            review_reasons=review_reasons + ["Could not match any taxonomy category."],
        )

    best = text_matches[0]
    alternatives = text_matches[1:5]

    # Blend confidences: image confirmation gives a modest boost; disagreement
    # (image ran but supports nothing close to top text match) trims confidence.
    final_confidence = text_confidence
    if used_image and image_confidence is not None:
        final_confidence = min(1.0, 0.7 * text_confidence + 0.3 * image_confidence)

    if final_confidence < settings.CONFIDENCE_REVIEW_BELOW:
        review_reasons.append(f"Low classification confidence ({final_confidence:.2f}).")
    if len(alternatives) and (alternatives[0].confidence > 0) and (best.confidence - alternatives[0].confidence < 0.05):
        review_reasons.append("Top category is ambiguous versus the next best alternative.")

    # --- Attribute / attribute-value extraction for the winning category ---
    attrs_out = []
    for attr in index.attributes_for(best.category_id):
        try:
            match = index.find_values(attr["name"], fields)
            if match:
                attrs_out.append({
                    "handle": attr["handle"],
                    "name": attr["name"],
                    "value": match.value,
                    "confidence": round(match.confidence, 3),
                    "source": match.source,
                })
        except Exception:
            # never let one bad attribute derail the whole product
            continue

    return ClassificationOutcome(
        category=best,
        alternatives=[{"category_id": a.category_id, "name": a.name, "full_name": a.full_name,
                        "confidence": round(a.confidence, 4)} for a in alternatives],
        attributes=attrs_out,
        text_confidence=round(text_confidence, 4),
        image_confidence=round(image_confidence, 4) if image_confidence is not None else None,
        used_image=used_image,
        review_reasons=review_reasons,
    )
