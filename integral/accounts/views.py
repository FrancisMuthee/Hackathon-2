from django.shortcuts import render
from django.http import HttpResponse
from django.core import mail

# Create your views here.
def home(request):
    return render(request, "home.html")



connection = mail.get_connection()

# Manually open the connection
connection.open()

# Construct an email message that uses the connection
email1 = mail.EmailMessage(
    "Hello",
    "Body goes here",
    "francisnjaramba2@gmail.com",
    ["reddmarx01@gmail.com"],
    connection=connection,
)
email1.send()  # Send the email