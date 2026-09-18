from django.urls import path

from . import api, views

urlpatterns = [
    path("", views.review_ui, name="review-ui"),
    path("api/products/", api.ProductListAPIView.as_view(), name="api-product-list"),
    path("api/products/<int:pk>/", api.ProductDetailAPIView.as_view(), name="api-product-detail"),
    path("api/products/<int:pk>/classification/", api.ClassificationUpdateAPIView.as_view(), name="api-classification-update"),
    path("api/import-batches/", api.ImportBatchListAPIView.as_view(), name="api-import-batches"),
    path("api/stats/", api.dashboard_stats, name="api-stats"),
    path("api/categories/search/", api.category_search, name="api-category-search"),
]
