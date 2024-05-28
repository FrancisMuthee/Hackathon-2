from django.shortcuts import render
from django.http import HttpResponse
from django.core.mail import EmailMessage, get_connection
from django.conf import settings
import os

# Create your views here.
def index(request):
    if request.method == 'POST':
        name = request.POST.get('full_name')
        email = request.POST.get('email')
        message = request.POST.get('message')

    #     connection= get_connection()
    #     connection.open()
    #     email1 = EmailMessage(
    #         "Hello",
    #         "Is everything OK?"
    #         "francisnjaramba2@gmail.com",
    #         ["reddmarx01@gmail.com"],
    #         connection=connection,
    #     )
    #     email1.send()
    #     connection.close()

    #     return HttpResponse('email sent successfully')
    # else:

    #     return render(request, 'home.html')


