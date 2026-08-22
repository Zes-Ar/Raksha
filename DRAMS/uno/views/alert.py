from django.core.mail import EmailMessage
from django.views.decorators.csrf import csrf_exempt
from uno.models import *
from twilio.rest import Client

from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse


twilio_account_sid = 'AC5ea8bb59991689451ceaf29ade8f320e'
twilio_auth_token = '7094c5b698252c3f9ab86103d2b22f23'
client = Client(twilio_account_sid, twilio_auth_token)


def alert_list(request, instance):
    if instance == "crowd":
        alertlist = Location.objects.filter(covid_status=1)
    elif instance == "flood":
        alertlist = Location.objects.filter(flood_status=1)
    elif instance == "fire":
        alertlist = Location.objects.filter(fire_status=1)
    if len(alertlist) == 0:
        alertlist = None
    return render(request, instance+'/alert.html', {'al': alertlist, 'instance': instance})


@csrf_exempt
def send_alert(request):
    med = request.POST.get("med")
    sub = request.POST.get("subject")
    body = request.POST.get("body")
    res = {"msg": None, "email": None}

    print(f"[ALERT] Medium: {med}, Subject: {sub}")

    users = User.objects.all()
    for user in users:
        if "e" in med and user.email is not None:
            try:
                email = EmailMessage(
                    sub, body, from_email="arpanchatterjee0101@gmail.com", to=[user.email]
                )
                email.send()
                print(f"[ALERT] Email sent to {user.email}")
                res["email"] = True
            except Exception as e:
                print(f"[ALERT] Email FAILED to {user.email}: {e}")
                res["email"] = False

        if "s" in med and user.phone is not None:
            try:
                message = client.messages.create(
                    body=body,
                    from_='+12057518316',
                    to='+91'+str(user.phone)
                )
                if message.error_code is None:
                    res["msg"] = True
                else:
                    res["msg"] = False
            except Exception as e:
                print(f"[ALERT] SMS FAILED to {user.phone}: {e}")
                res["msg"] = False

    return JsonResponse(res)
