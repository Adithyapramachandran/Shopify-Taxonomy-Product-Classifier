from django.contrib import admin

from .models import Classification, ImportBatch, ProcessingError, Product


@admin.register(ImportBatch)
class ImportBatchAdmin(admin.ModelAdmin):
    list_display = ("id", "source_filename", "status", "total_rows", "imported_rows", "failed_rows", "created_at")


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("sku", "title", "source_category", "has_description", "has_image")
    search_fields = ("sku", "title", "description")
    list_filter = ("has_description", "has_image", "source_category")


@admin.register(Classification)
class ClassificationAdmin(admin.ModelAdmin):
    list_display = ("product", "category_name", "confidence", "status", "used_image")
    list_filter = ("status", "used_image")
    search_fields = ("product__sku", "product__title", "category_name")


@admin.register(ProcessingError)
class ProcessingErrorAdmin(admin.ModelAdmin):
    list_display = ("stage", "row_reference", "message", "created_at")
    list_filter = ("stage",)
