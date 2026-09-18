from rest_framework import serializers

from .models import Classification, ImportBatch, Product


class ClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Classification
        fields = [
            "id", "category_id", "category_name", "category_full_name",
            "confidence", "text_confidence", "image_confidence", "used_image",
            "alternatives", "attributes", "status", "review_reasons",
            "engine_version", "reviewed_by", "reviewed_at", "reviewer_notes",
            "updated_at",
        ]
        read_only_fields = ["engine_version", "updated_at"]


class ProductListSerializer(serializers.ModelSerializer):
    classification = ClassificationSerializer(read_only=True)

    class Meta:
        model = Product
        fields = [
            "id", "sku", "title", "brand", "source_category", "source_sub_category",
            "has_description", "has_image", "image_urls", "classification",
        ]


class ProductDetailSerializer(serializers.ModelSerializer):
    classification = ClassificationSerializer(read_only=True)

    class Meta:
        model = Product
        fields = "__all__"


class ImportBatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImportBatch
        fields = "__all__"


class ClassificationUpdateSerializer(serializers.ModelSerializer):
    """Used by the reviewer to approve / correct a classification."""

    class Meta:
        model = Classification
        fields = ["category_id", "category_name", "category_full_name",
                   "attributes", "status", "reviewer_notes"]
