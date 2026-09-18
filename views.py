from django.shortcuts import render


def review_ui(request):
    """Single-page review interface (HTML shell; data loads via the JSON API)."""
    return render(request, "classifier/review.html")
