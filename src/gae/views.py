from django.shortcuts import render


def dash_update(request, dash):
    # This view should pass on the redirect
    template = 'home.html'
    print(dash)
    context = {
        "dash": dash,
    }
    return render(request, template, context)
