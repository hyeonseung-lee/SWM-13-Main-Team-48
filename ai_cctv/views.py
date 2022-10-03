from django.shortcuts import render


def main(request):
    return render(request, 'main.html')


def analytics(request):
    return render(request, 'analytics.html')
