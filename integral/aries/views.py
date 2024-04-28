from django.shortcuts import render
from django.http import HttpResponse
from django.core.mail import send_mail
from django.conf import settings
import os

# Create your views here.
def index(request):
    if request.method == 'POST':
        name = request.POST.get('full_name')
        email = request.POST.get('email')
        message = request.POST.get('message')

        data = {
            'name': name,
            'email': email,
            'message': message,
        }

        # Constructing the email message
        message_content = f"""
        New message: {data['message']}

        From: {data['email']}
        """

        # Sending the email
        send_mail(
            'New message from your website', # Subject of the email
            message_content, # Message body
            settings.EMAIL_HOST_USER, # From email
            ['reddmarx01@gmail.com'], # To email
            fail_silently=False,
        )

        # Optionally, you can redirect to a success page or show a success message
        # return HttpResponse('Message sent successfully')
        return render(request, 'index.html', {'message_sent': True})

    return render(request, 'index.html', {})

