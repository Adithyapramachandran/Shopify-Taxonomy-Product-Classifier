from django.db.models import Avg, Count, Q
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .engine import get_taxonomy_index
from .models import Classification, ImportBatch, Product
from .serializers import (
    ClassificationUpdateSerializer,
    ImportBatchSerializer,
    ProductDetailSerializer,
    ProductListSerializer,
)


class ProductListAPIView(generics.ListAPIView):
    """
    GET /api/products/?status=needs_review&min_confidence=0&max_confidence=0.45&q=sofa&category=&has_image=1
    """
    serializer_class = ProductListSerializer

    def get_queryset(self):
        qs = Product.objects.select_related("classification").all().order_by("-id")
        params = self.request.query_params

        status_filter = params.get("status")
        if status_filter:
            qs = qs.filter(classification__status=status_filter)

        q = params.get("q")
        if q:
            qs = qs.filter(Q(title__icontains=q) | Q(sku__icontains=q) | Q(description__icontains=q))

        min_conf = params.get("min_confidence")
        if min_conf is not None:
            qs = qs.filter(classification__confidence__gte=float(min_conf))
        max_conf = params.get("max_confidence")
        if max_conf is not None:
            qs = qs.filter(classification__confidence__lte=float(max_conf))

        category = params.get("category")
        if category:
            qs = qs.filter(classification__category_name__icontains=category)

        has_image = params.get("has_image")
        if has_image is not None:
            qs = qs.filter(has_image=(has_image == "1"))

        import_batch = params.get("import_batch")
        if import_batch:
            qs = qs.filter(import_batch_id=import_batch)

        return qs


class ProductDetailAPIView(generics.RetrieveAPIView):
    queryset = Product.objects.select_related("classification").all()
    serializer_class = ProductDetailSerializer


class ClassificationUpdateAPIView(APIView):
    """PATCH /api/products/<id>/classification/  -- approve or correct a result."""

    def patch(self, request, pk):
        try:
            product = Product.objects.select_related("classification").get(pk=pk)
        except Product.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)

        classification, _ = Classification.objects.get_or_create(product=product)
        serializer = ClassificationUpdateSerializer(classification, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        classification = serializer.save()

        if request.data.get("status") in ("approved", "rejected"):
            classification.reviewed_at = timezone.now()
            if request.user and request.user.is_authenticated:
                classification.reviewed_by = request.user
            classification.save()

        return Response(ClassificationUpdateSerializer(classification).data)


class ImportBatchListAPIView(generics.ListAPIView):
    queryset = ImportBatch.objects.all().order_by("-id")
    serializer_class = ImportBatchSerializer


@api_view(["GET"])
def dashboard_stats(request):
    total = Product.objects.count()
    classified = Classification.objects.count()
    by_status = {
        row["status"]: row["n"]
        for row in Classification.objects.values("status").annotate(n=Count("id"))
    }
    avg_conf = Classification.objects.aggregate(a=Avg("confidence"))["a"] or 0
    no_image = Product.objects.filter(has_image=False).count()
    no_description = Product.objects.filter(has_description=False).count()
    return Response({
        "total_products": total,
        "classified": classified,
        "unclassified": total - classified,
        "by_status": by_status,
        "average_confidence": round(avg_conf, 3),
        "products_without_image": no_image,
        "products_without_description": no_description,
    })


@api_view(["GET"])
def category_search(request):
    """GET /api/categories/search/?q=sofa  -- used by the review UI's category picker."""
    q = request.query_params.get("q", "")
    if len(q) < 2:
        return Response([])
    index = get_taxonomy_index()
    matches = index.search(q, top_n=10)
    return Response([
        {"category_id": m.category_id, "name": m.name, "full_name": m.full_name}
        for m in matches
    ])
