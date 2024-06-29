from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from app.verify import authentication   #, form_varification
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
import os
import cv2
import numpy as np
from tensorflow import keras
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import HttpResponse
from io import BytesIO
from .models import *
from .process import *


# Create your views here.
def index(request):
    # return HttpResponse("This is Home page")    
    return render(request, "index.html", {'action' : 'index'})

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html", {'action' : 'log_in'})

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html", {'action' : 'register'})

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname': request.user.first_name, 
        # 'form' : patient_form(),
        }
    if request.method == "POST":
        net = cv2.dnn.readNet("models/yolov3_training_2000.weights", "models/yolov3_testing.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        classes = ["Weapon","Knife"]



        cap = cv2.VideoCapture(0)

        while True:
            _, img = cap.read()
            # height, width, channels = img.shape
            width = 512
            height = 512
            channels = 3

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            
            layer_names = net.getLayerNames()

            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            outs = net.forward(output_layers)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0: print("weapon detected in frame")
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

            # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Weapon Detection", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return redirect("dashboard")
    
    return render(request, "dashboard.html", context)


@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def crime_detection(request):
    context = {'fname': request.user.first_name}

    if request.method == "POST":
        uploaded_file = request.FILES['video']
        upload_video = UploadedVideo.objects.create(video = uploaded_file)
        upload_video.save()
        messages.success(request, "Predicted Successfuly!!")
        return redirect('predict')
    
    return render(request, "crime_detection.html", context)


@login_required(login_url="log_in")
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def predict(request):
    upload_video = UploadedVideo.objects.last()
    path = str(upload_video.video.url)
    print(path)
    path = path[1:]
    print(path)
    predicted_class = predict_video_class(path)

    # Print the predicted class
    print("Predicted class:", predicted_class)
    context = {
        'fname': request.user.first_name,
        'upload_video' : upload_video,
        'predicted_class' : predicted_class
        }
    
    return render(request, "predict.html", context)