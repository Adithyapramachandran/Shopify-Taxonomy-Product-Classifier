from django.db import models
from django.contrib.auth.models import User


class ImportBatch(models.Model):
    """One upload/import run of a product spreadsheet."""
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    source_filename = models.CharField(max_length=512)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    total_rows = models.IntegerField(default=0)
    imported_rows = models.IntegerField(default=0)
    failed_rows = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error_log = models.TextField(blank=True, default="")

    def __str__(self):
        return f"Batch {self.id}: {self.source_filename} ({self.status})"


class Product(models.Model):
    """A single product row imported from the catalogue."""

    import_batch = models.ForeignKey(ImportBatch, on_delete=models.SET_NULL, null=True, related_name="products")

    sku = models.CharField(max_length=128, unique=True, db_index=True)
    model_number = models.CharField(max_length=128, blank=True, default="")

    title = models.CharField(max_length=1024, blank=True, default="")
    description = models.TextField(blank=True, default="")
    bullets = models.TextField(blank=True, default="")

    brand = models.CharField(max_length=255, blank=True, default="")
    source_category = models.CharField(max_length=255, blank=True, default="")
    source_sub_category = models.CharField(max_length=255, blank=True, default="")
    collection_name = models.CharField(max_length=255, blank=True, default="")
    color = models.CharField(max_length=255, blank=True, default="")
    materials = models.CharField(max_length=512, blank=True, default="")
    dimensions = models.CharField(max_length=512, blank=True, default="")
    weight = models.CharField(max_length=64, blank=True, default="")
    country_of_origin = models.CharField(max_length=128, blank=True, default="")

    image_urls = models.JSONField(default=list, blank=True)
    product_url = models.URLField(max_length=1024, blank=True, default="")

    raw_data = models.JSONField(default=dict, blank=True)

    has_description = models.BooleanField(default=False)
    has_image = models.BooleanField(default=False)
    data_quality_notes = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.sku} - {self.title[:60]}"


class Classification(models.Model):
    """The (current) classification result for a product. One-to-one with Product."""

    STATUS_CHOICES = [
        ("pending", "Pending review"),
        ("auto_approved", "Auto-approved (high confidence)"),
        ("needs_review", "Needs manual review"),
        ("approved", "Approved by reviewer"),
        ("rejected", "Rejected by reviewer"),
        ("error", "Classification failed"),
    ]

    product = models.OneToOneField(Product, on_delete=models.CASCADE, related_name="classification")

    category_id = models.CharField(max_length=128, blank=True, default="")
    category_name = models.CharField(max_length=255, blank=True, default="")
    category_full_name = models.CharField(max_length=512, blank=True, default="")

    confidence = models.FloatField(default=0.0)
    text_confidence = models.FloatField(null=True, blank=True)
    image_confidence = models.FloatField(null=True, blank=True)
    used_image = models.BooleanField(default=False)

    alternatives = models.JSONField(default=list, blank=True)
    attributes = models.JSONField(default=list, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    review_reasons = models.JSONField(default=list, blank=True)
    engine_version = models.CharField(max_length=32, blank=True, default="")

    reviewed_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewer_notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.product.sku} -> {self.category_name} ({self.confidence:.2f})"


class ProcessingError(models.Model):
    """Every failure is logged here instead of stopping the batch (requirement #11)."""

    import_batch = models.ForeignKey(ImportBatch, on_delete=models.CASCADE, related_name="processing_errors", null=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="errors", null=True, blank=True)
    stage = models.CharField(max_length=32)
    row_reference = models.CharField(max_length=128, blank=True, default="")
    message = models.TextField()
    traceback = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.stage}] {self.row_reference}: {self.message[:80]}"
